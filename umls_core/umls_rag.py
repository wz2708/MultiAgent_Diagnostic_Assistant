import sqlite3, json, re, math, sys
from collections import defaultdict, Counter
try:
    from rapidfuzz import fuzz, process as rf_process
    RAPID_OK = True
except Exception:
    RAPID_OK = False


USE_SAPBERT = True
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

def _strip_semantic_tag(s: str) -> str:
    return re.sub(r"\s*\((?:finding|disorder|morphologic abnormality|symptom|sign|procedure|qualifier value)\)\s*$","",s,flags=re.I)

HEAD_NOUNS = {"pain","pains","ache","aches","tenderness","swelling","erythema","edema",
              "itching","pruritus","weakness","numbness","stiffness","spasm","spasms",
              "tremor","tremors","paralysis","paresis","soreness"}
PLURAL_TO_SING = {"pains":"pain","aches":"ache","spasms":"spasm","tremors":"tremor"}

def canon_text(s: str) -> str:
    '''
        Canonicalize a symptom/disease phrase by:
            Lowercasing, removing punctuation & stop suffixes (e.g., “pain in chest” → “chest pain”).
            Handling plural forms (pains → pain).
            Stripping UMLS-style artifacts.
        Output: Clean, normalized string for indexing/matching
    '''
    if not s: return ""
    s = _strip_semantic_tag(s)
    s = re.sub(r"[^\w\s,;/\-\(\)\[\]]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().replace("–","-").replace("—","-")
    s_low = s.lower()
    s_low = re.sub(r"^pain\s+in\s+(.+)$", r"\1 pain", s_low)
    s_low = re.sub(r"^pains?\s+[,\s]*([a-z].+)$", r"\1 pain", s_low)
    s_low = re.sub(r"^(.+?)[,;]\s*pain$", r"\1 pain", s_low)
    toks = s_low.split()
    if toks and toks[0] in HEAD_NOUNS and len(toks)>=2:
        head = PLURAL_TO_SING.get(toks[0], toks[0])
        s_low = " ".join(toks[1:] + [head])
    s_low = re.sub(r"[;\[\]\(\)]"," ", s_low)
    s_low = re.sub(r"[^a-z\s]"," ", s_low)
    s_low = re.sub(r"\s+"," ", s_low).strip()
    words = s_low.split()
    if words and words[-1] in PLURAL_TO_SING:
        words[-1] = PLURAL_TO_SING[words[-1]]
    return " ".join(words)

class EnhancedWhitelistRAG:
    '''
        Inputs:
            mini_db_path: path to SQLite file with disease/symptom mappings.
            fuzzy_k: number of fuzzy candidates per query term.
            min_hits: min number of symptoms required to consider a disease.
            fuzzy_thr: similarity threshold (0–100).
            dense_weight: weight of dense SapBERT embeddings.

        Builds:
            dis_info: {cui → {name, definition, codes}}
            cui2sym: {cui → set(symptoms)}
            sym2cui: {symptom → set(cui)}
            idf: IDF-style weights for symptoms.
            Dense SapBERT embeddings (optional, if USE_SAPBERT=True).
    '''
    def __init__(self, mini_db_path: str, fuzzy_k=20, min_hits=1, fuzzy_thr=75, dense_weight=0.6):
        self.db = sqlite3.connect(mini_db_path)
        self.db.row_factory = sqlite3.Row
        self.min_hits = min_hits
        self.dis_info = {}                 # cui -> {name, definition, codes}
        self.cui2sym = defaultdict(set)    # cui -> set(symptom_text)
        self.name2cui = {}                 # lower(name) -> cui
        vocab = set()

        for r in self.db.execute("SELECT cui,name,definition,codes FROM diseases"):
            codes = json.loads(r["codes"] or "{}")
            self.dis_info[r["cui"]] = {"name": r["name"], "definition": r["definition"], "codes": codes}
            self.name2cui[r["name"].lower()] = r["cui"]

        for r in self.db.execute("SELECT disease_cui,symptom FROM disease_symptoms"):
            if r["symptom"]:
                self.cui2sym[r["disease_cui"]].add(r["symptom"])
                vocab.add(r["symptom"])

        self.sym2cui = defaultdict(set)
        for cui, ss in self.cui2sym.items():
            for s in ss:
                self.sym2cui[s].add(cui)

        N = max(1, len(self.dis_info))
        self.idf = {}
        for s, S in self.sym2cui.items():
            df = max(1, len(S))
            self.idf[s] = math.log((N+0.5) / df)

        self.vocab = sorted(vocab)
        self.fuzzy_k = fuzzy_k
        self.fuzzy_thr = fuzzy_thr

        self.dense_weight = dense_weight
        self._dense_ready = False
        if USE_SAPBERT:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                self._tok = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
                self._mdl = AutoModel.from_pretrained(SAPBERT_MODEL)
                self._mdl.eval()
                self._torch = torch
                self.cui2doc = {}
                for cui, ss in self.cui2sym.items():
                    if not ss:
                        self.cui2doc[cui] = self.dis_info[cui]["name"]
                        continue
                    top_sym = sorted(ss, key=lambda x: self.idf.get(x, 0.0), reverse=True)[:20]
                    self.cui2doc[cui] = self.dis_info[cui]["name"] + " | " + "; ".join(top_sym)
                with self._torch.no_grad():
                    embs = []
                    self._cuis = list(self.cui2doc.keys())
                    for i in range(0, len(self._cuis), 32):
                        batch_cuis = self._cuis[i:i+32]
                        batch_txt = [self.cui2doc[c] for c in batch_cuis]
                        tok = self._tok(batch_txt, padding=True, truncation=True, max_length=128, return_tensors="pt")
                        out = self._mdl(**tok)
                        emb = out.last_hidden_state.mean(dim=1)   # mean pooling
                        emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
                        embs.append(emb)
                    self._D = self._torch.cat(embs, dim=0)  # [M, H]
                self._dense_ready = True
            except Exception:
                self._dense_ready = False

    def _dense_score(self, query_terms):
        '''
            Private Method
            Compute dense similarity scores between query terms and diseases using SapBERT embeddings.
            Output: {cui → similarity score × dense_weight}.
        '''
        if not self._dense_ready:
            return {}
        qtxt = " | ".join(sorted(set(query_terms)))
        tok = self._tok([qtxt], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with self._torch.no_grad():
            out = self._mdl(**tok)
            q = out.last_hidden_state.mean(dim=1)
            q = q / (q.norm(dim=1, keepdim=True) + 1e-8)        # [1, H]
            sims = (q @ self._D.T).squeeze(0)                   # [M]
            vals = sims.detach().cpu().tolist()
        dense = {}
        for i, cui in enumerate(self._cuis):
            dense[cui] = max(0.0, float(vals[i])) * self.dense_weight
        return dense

    def symptoms_to_diseases(self, symptom_phrases, topk=10, topk_symptoms=6):
        '''
        Main retrieval function.
            Normalize symptoms with canon_text.
            Steps:
                Exact match scores: weighted by IDF.
                Fuzzy match scores: RapidFuzz token similarity.
                Dense scores: SapBERT semantic similarity.
                Merge scores and rank diseases.
            For each disease:
                Return matched, typical, and missing key symptoms.
                Include definition and standard codes.
                Output: {"candidates": [ ... ]}
        '''
        q_terms_raw = [canon_text(x) for x in symptom_phrases if canon_text(x)]
        if not q_terms_raw:
            return {"candidates": []}
        q_terms = sorted(set(q_terms_raw))

        exact_score = defaultdict(float)
        for t in q_terms:
            if t in self.sym2cui:
                idf = self.idf.get(t, 0.0)
                for cui in self.sym2cui[t]:
                    exact_score[cui] += 2.0 * idf
        fuzzy_score = defaultdict(float)
        matched_map = defaultdict(set)  # cui -> set(matched terms)
        if RAPID_OK and self.vocab:
            for qt in q_terms:
                if qt in self.sym2cui:
                    for cui in self.sym2cui[qt]:
                        matched_map[cui].add(qt)
                    continue
                cand = rf_process.extract(qt, self.vocab, scorer=fuzz.token_set_ratio, limit=self.fuzzy_k)
                for item in cand:
                    sym = item[0]
                    sim = item[1]
                    if sim < self.fuzzy_thr:
                        continue
                    idf = self.idf.get(sym, 0.0)
                    for cui in self.sym2cui.get(sym, ()):
                        fuzzy_score[cui] += (sim / 100.0) * idf
                        matched_map[cui].add(sym)

        dense_score = self._dense_score(q_terms) if self._dense_ready else {}

        all_cuis = set(exact_score) | set(fuzzy_score) | set(dense_score)
        scored = []
        for cui in all_cuis:
            s = exact_score.get(cui, 0.0) + fuzzy_score.get(cui, 0.0) + dense_score.get(cui, 0.0)
            scored.append((cui, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        items = []
        for cui, sc in scored[:max(topk*2, 40)]:
            info = self.dis_info.get(cui)
            if not info:
                continue
            syms = list(self.cui2sym.get(cui, set()))
            syms_sorted = sorted(syms, key=lambda x: self.idf.get(x, 0.0), reverse=True)
            matched = [t for t in syms_sorted if t in matched_map.get(cui, set())][:topk_symptoms]

            typical = syms_sorted[:max(topk_symptoms, 8)]
            missing = [t for t in typical if t not in matched][:max(0, topk_symptoms - len(matched))]
            items.append({
                "name": info["name"],
                "definition": info["definition"],
                "codes": info["codes"],
                "matched_symptoms": matched,
                "typical_symptoms": typical[:topk_symptoms],
                "missing_key_symptoms": missing[:topk_symptoms],
                "score": round(sc, 3)
            })

        out = [x for x in items if (x["matched_symptoms"] or x["score"] >= 0.2)]
        return {"candidates": out[:topk]}

    def diseases_to_info(self, diseases, topk_symptoms=8):
        '''
            Map given disease names to their definitions, codes, and typical symptoms.
                Name is looked up in DB (name2cui).
                Fallback: substring match.
            Output: {"items": [ ... ]}
        '''
        items=[]
        for d in diseases:
            cui = self.name2cui.get(d.lower())
            if not cui:
                for k,v in self.name2cui.items():
                    if d.lower() in k:
                        cui = v; break
            if not cui:
                continue
            info = self.dis_info[cui]
            syms = list(self.cui2sym.get(cui, set()))
            syms_sorted = sorted(syms, key=lambda x: self.idf.get(x, 0.0), reverse=True)
            items.append({
                "name": info["name"],
                "definition": info["definition"],
                "codes": info["codes"],
                "typical_symptoms": syms_sorted[:topk_symptoms]
            })
        return {"items": items}


# if __name__ == "__main__":
#     dbp = sys.argv[1] if len(sys.argv)>1 else "umls_out/mini_kit.sqlite"
#     rag = EnhancedWhitelistRAG(dbp, min_hits=2)
#     symptoms = ["back pain","productive cough","limb weakness","sore neck","dizzy","off balance"]
#     print(json.dumps(rag.symptoms_to_diseases(symptoms, topk=3), indent=2))
#     print(json.dumps(rag.diseases_to_info(["pneumonia"], topk_symptoms=8), indent=2))
