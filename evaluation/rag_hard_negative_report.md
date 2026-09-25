# Offline RAG Evaluation

> Deterministic keyword retrieval only. This is not an online LLM answer-quality score.

- Cases: 38
- Answerable / empty-retrieval / generation-only: 24 / 6 / 8
- Hit@1: 100.0%
- Recall@3: 100.0%
- MRR: 1.000
- Empty-retrieval negative accuracy: 100.0%
- False-positive rate: 0.0%
- Graded case accuracy: 100.0%
- In-domain unknowns receiving context: 87.5%
- No-result rate: 18.4%
- Citation render integrity (not answer faithfulness): 100.0%
- By difficulty: adversarial 5/5 graded, in_domain_unknown 8 ungraded, multi_step 8/8 graded, no_answer 6/6 graded, normal 7/7 graded, paraphrase 4/4 graded

| Case | Difficulty | Expected | Retrieved | Pass | RR |
|---|---|---|---|---:|---:|
| rag-01 | normal | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-02 | normal | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-03 | multi_step | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-04 | normal | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-05 | normal | transport_and_pacing | transport_and_pacing | yes | 1.000 |
| rag-06 | normal | chongqing_visitor_guide, transport_and_pacing | transport_and_pacing, chongqing_visitor_guide | yes | 1.000 |
| rag-07 | multi_step | transport_and_pacing | transport_and_pacing, chongqing_visitor_guide, responsible_trip_checklist | yes | 1.000 |
| rag-08 | multi_step | transport_and_pacing | transport_and_pacing | yes | 1.000 |
| rag-09 | adversarial | responsible_trip_checklist | responsible_trip_checklist | yes | 1.000 |
| rag-10 | adversarial | responsible_trip_checklist | responsible_trip_checklist | yes | 1.000 |
| rag-11 | multi_step | responsible_trip_checklist | responsible_trip_checklist | yes | 1.000 |
| rag-12 | normal | responsible_trip_checklist | responsible_trip_checklist, chongqing_visitor_guide, transport_and_pacing | yes | 1.000 |
| rag-13 | paraphrase | chongqing_visitor_guide | chongqing_visitor_guide, transport_and_pacing | yes | 1.000 |
| rag-14 | paraphrase | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-15 | multi_step | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-16 | adversarial | chongqing_visitor_guide, transport_and_pacing | chongqing_visitor_guide, transport_and_pacing | yes | 1.000 |
| rag-17 | multi_step | chongqing_visitor_guide | chongqing_visitor_guide | yes | 1.000 |
| rag-18 | paraphrase | transport_and_pacing | transport_and_pacing, chongqing_visitor_guide, responsible_trip_checklist | yes | 1.000 |
| rag-19 | normal | transport_and_pacing | transport_and_pacing, chongqing_visitor_guide | yes | 1.000 |
| rag-20 | multi_step | transport_and_pacing | transport_and_pacing, chongqing_visitor_guide | yes | 1.000 |
| rag-21 | paraphrase | responsible_trip_checklist | responsible_trip_checklist | yes | 1.000 |
| rag-22 | adversarial | responsible_trip_checklist | responsible_trip_checklist | yes | 1.000 |
| rag-23 | adversarial | responsible_trip_checklist | responsible_trip_checklist, chongqing_visitor_guide | yes | 1.000 |
| rag-24 | multi_step | chongqing_visitor_guide, responsible_trip_checklist | responsible_trip_checklist, chongqing_visitor_guide | yes | 1.000 |
| rag-25 | no_answer |  | - | yes | 0.000 |
| rag-26 | no_answer |  | - | yes | 0.000 |
| rag-27 | no_answer |  | - | yes | 0.000 |
| rag-28 | no_answer |  | - | yes | 0.000 |
| rag-29 | no_answer |  | - | yes | 0.000 |
| rag-30 | no_answer |  | - | yes | 0.000 |
| rag-hard-01 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-02 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-03 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-04 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-05 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-06 | in_domain_unknown |  | - | ungraded | 0.000 |
| rag-hard-07 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |
| rag-hard-08 | in_domain_unknown |  | chongqing_visitor_guide | ungraded | 0.000 |

Boundary: No LLM generation is graded. In-domain unknowns need generated-answer abstention or clarification checks and are ungraded here. Citation render integrity checks the evaluator's own formatting, not model faithfulness.
