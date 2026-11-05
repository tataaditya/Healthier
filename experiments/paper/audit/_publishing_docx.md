
Application of Deep Learning for Nutritional Segmentation and Information Extraction in the Healthier Application

Tata Aditya Pamungkas1, Nuke Lu’lu Ul Chusna2, Harry Dwiyana Kartika3

1,2,3 Informatics Engineering
Universitas Krisnadwipayana, Jakarta, Indonesia
tataaditaya@gmail.com, nukelchusna@unkris.ac.id, harry_d@unkris.ac.id



Abstract
Low nutritional literacy and the high risk of Non-Communicable Diseases (NCDs) have driven the development of Healthier, an AI-based nutrition analysis application integrating food recognition from plate images with nutrition-label extraction from packaging. Food segmentation uses a YOLO11s-seg model trained on 19,087 images, while nutrition-text extraction relies on PaddleOCR PP-OCRv5 combined with Fuzzy Matching. The extracted data are processed by a Clinical Rules Engine based on PERKENI 2024 and JNC 8 guidelines to generate personalized recommendations. The segmentation model achieved a Precision of 72.07%, Recall of 60.46%, Mask mAP50 of 67.06%, and mAP50-95 of 52.84%, with an inference time of 168 ms per image, while the OCR module achieved a Field-Level Accuracy of 83.8%. A Human-in-the-Loop mechanism allows users to manually verify inference results. Overall, this evaluation confirms Healthier's potential as a practical nutrition-assistance system that combines visual analysis and health recommendations, offering a low-cost, edge-based alternative to cloud-dependent nutrition-labelling tools for patients managing diabetes and hypertension.

Keywords: Instance Segmentation; OCR; YOLO11s-seg; PaddleOCR; Nutrition Literacy


Abstrak
Literasi gizi yang rendah dan tingginya risiko Penyakit Tidak Menular (PTM) mendorong pengembangan Healthier, sebuah aplikasi analisis gizi berbasis AI yang mengintegrasikan pengenalan makanan dari citra piring dengan ekstraksi label gizi dari kemasan. Segmentasi makanan menggunakan model YOLO11s-seg yang dilatih pada 19.087 citra, sedangkan ekstraksi teks gizi memanfaatkan PaddleOCR PP-OCRv5 yang dipadukan dengan Fuzzy Matching. Data hasil ekstraksi diproses oleh Clinical Rules Engine berbasis pedoman PERKENI 2024 dan JNC 8 untuk menghasilkan rekomendasi yang dipersonalisasi. Model segmentasi mencapai Precision 72,07%, Recall 60,46%, Mask mAP50 67,06%, dan mAP50-95 52,84%, dengan waktu inferensi 168 ms per citra, sementara modul OCR mencapai Field-Level Accuracy sebesar 83,8%. Mekanisme Human-in-the-Loop memungkinkan pengguna memverifikasi hasil inferensi secara manual. Secara keseluruhan, evaluasi ini menegaskan potensi Healthier sebagai sistem bantu gizi praktis yang memadukan analisis visual dan rekomendasi kesehatan, serta menawarkan alternatif berbiaya rendah berbasis edge computing dibandingkan perangkat pelabelan gizi yang bergantung pada cloud, bagi pasien penderita diabetes dan hipertensi.

Kata kunci: Segmentasi Instan; OCR; YOLO11s-seg; PaddleOCR; Literasi Gizi


INTRODUCTION

Non-communicable diseases (NCDs) remain closely associated with dietary patterns and other modifiable risk factors, with diabetes representing a substantial and growing burden in Indonesia. Recent projections indicate a continued increase in diabetes morbidity through 2045, reinforcing the need for practical prevention and self-management tools (Wahidin et al., 2024). At the individual level, the ability to interpret nutrition information is also limited. A study of 1,029 adults in Semarang found that 96.02% of respondents were classified as having problematic nutrition-fact literacy (Aprianti Aprianti et al., 2023). These findings support the need for tools that reduce the cognitive and practical burden of interpreting everyday food information.
Nutrition labelling can make food composition more accessible, particularly when information is presented in a simplified front-of-pack format. Recent systematic reviews report that interpretive front-of-pack labels can improve consumers' understanding of nutritional quality, although effects on actual purchasing and dietary intake vary by setting (Ganderats-Fuentes & Morgan, 2023; Shrestha et al., 2023). Nutri-Score and related interpretive schemes have therefore become important references in nutrition-information design(Bonaccio et al., 2023) . In practice, however, conventional nutrition tables remain difficult to process automatically because text can be small, layouts vary across products, and reflections or curved packaging can degrade image quality. For meals served on a plate, the problem is different: there is no printed nutrition table to scan, so dietary assessment must begin from visual food recognition (Dalakleidi et al., 2022; Mohanty et al., 2022).
This study addresses these two complementary problems: (1) estimating food composition from plate images through instance segmentation and (2) extracting nutrition-fact values from packaged-food labels through optical character recognition (OCR). Both outputs are processed through a common decision layer intended to support dietary management for users concerned with diabetes and hypertension. The scope is limited to Indonesian plate-based dishes grouped according to the Isi Piringku concept and to packaged-product nutrition tables commonly encountered in Indonesian retail settings. Recent reviews show that image-based dietary assessment typically involves food detection or segmentation, recognition, portion or volume estimation, and subsequent nutrient estimation; they also identify dataset diversity, portion estimation, and real-world image conditions as persistent challenges (Chotwanvirat et al., 2024; Dalakleidi et al., 2022; Shonkoff et al., 2023).
Accordingly, this research develops and evaluates Healthier, an AI-based nutrition-assistance application that integrates YOLO11s-seg food segmentation with PaddleOCR PP-OCRv5-based label extraction. The extracted information is passed to a deterministic Clinical Rules Engine aligned with current diabetes and hypertension nutrition guidance, with user verification retained as an explicit control point. Rather than attempting to replace professional dietary assessment, the system is designed as a practical decision-support layer that makes visual and label-based nutrition information easier to interpret.
Previous work demonstrates the feasibility of computer vision for dietary assessment, but the literature remains fragmented across individual stages of the pipeline. Large-scale benchmarks have shown that deep learning can recognize diverse foods from mobile images (Mohanty et al., 2022), while systematic reviews report that segmentation, recognition, and nutrient estimation are commonly developed as separate tasks (Chotwanvirat et al., 2024; Dalakleidi et al., 2022; Shonkoff et al., 2023). Recent models have explored transformer-based recognition (Xiao et al., n.d.), lightweight food recognition(Wang et al., 2024), and segmentation-based nutrient estimation (Zhao et al., 2024). Indonesian studies have also evaluated YOLO-based food detection and nutritional estimation in local-food settings (Hananta & Cahyono, 2026; Utami et al., 2023). Food segmentation itself remains challenging when multiple foods have similar appearance or overlap (Raju et al., 2023), and the reliability of downstream nutrition estimation is strongly affected by the quality of portion or volume estimation (Shonkoff et al., 2023; Zhao et al., 2024).
The label-reading problem presents a parallel challenge. Conventional OCR pipelines can be affected by packaging geometry, reflections, and inconsistent text layouts, motivating the use of preprocessing, layout reconstruction, and domain-specific correction(Rosyadi et al., 2024; Saputra et al., 2024). More recent dietary-assessment studies also show that manual correction or user input remains valuable when automated recognition encounters ambiguous foods or portion estimates (Folson et al., 2023; Moyen et al., 2022; Oei et al., 2024). This motivates the Human-in-the-Loop (HITL) design adopted in Healthier, in which users can verify segmentation and extracted nutrient values before they are used by the decision layer (Mosqueira-Rey et al., 2023; Yuan et al., 2024).
The research gap therefore lies not simply in improving one recognition model, but in integrating food segmentation, nutrition-label extraction, clinical-rule processing, and human verification into a single local workflow. Existing studies provide strong evidence for individual components, yet relatively few combine plate-based food analysis with packaged-label extraction and deterministic nutrition rules while retaining an offline or edge-oriented deployment strategy. Edge-AI research highlights the value of moving computation closer to the user when device resources, latency, connectivity, or data-privacy constraints make continuous cloud processing undesirable. Healthier addresses this integration gap through an edge-oriented pipeline that combines YOLO11s-seg, PaddleOCR, deterministic rules, and HITL verification. The research question is therefore: how can food composition and packaged-food nutrition information be extracted locally and translated into transparent, guideline-aligned dietary feedback while preserving an explicit opportunity for user verification?

RESEARCH METHODS

This research adopts an iterative experimental methodology informed by CRISP-ML(Q), with separate evaluation of the segmentation, OCR, clinical-rule, and usability components. The methodology emphasizes repeated validation because recent dietary-assessment reviews have identified variability in food images, datasets, ground truth, and reporting practices as major barriers to reliable comparison between systems (Chotwanvirat et al., 2024; Dalakleidi et al., 2022; Shonkoff et al., 2023) . All experiments were conducted on a laptop equipped with a 12th-generation Intel Core i5 processor, 8 GB RAM, and an NVIDIA RTX 3050 GPU with 6 GB GDDR6 VRAM.

Segmentation Model Development
The YOLO11s-seg model was trained for 50 epochs using AdamW optimization. Spatial and photometric augmentation, including HSV adjustment, mosaic, rotation, and horizontal flipping, was applied to improve robustness. Instance or semantic segmentation is a central component of image-based dietary assessment because separating food regions provides the spatial information required for subsequent recognition and portion analysis (Dalakleidi et al., 2022; Raju et al., 2023; Zhu & Dai, 2023). Recent work has further demonstrated that segmentation can be combined with nutrient regression to estimate food mass and nutrient content (Zhao et al., 2024). The YOLO11 architecture was selected for its computational efficiency and suitability for edge-oriented inference.

Table 1. Dataset distribution and characteristics
Dataset Source
Number of Images
Access
Category
UECFoodPIXCOMPLETE
10,000
Public
Segmentation
FruitSeg30
1,969
Public
Segmentation
FoodSeg103
7,118
Public
Segmentation
Supermarket Nutrition Table
446
Private
OCR
Total
19,533
—
—
Class labels were remapped into five balanced-nutrition guideline clusters: Staple Food, Side Dishes, Vegetables, Fruits, and Others. The dataset was split with an 80:20 ratio (training and validation) with no data overlap. Dynamic spatial and photometric augmentation parameters (hsv, mosaic, fliplr) were applied to reduce overfitting. The visual nutrition-proportion calculation compares the ratio of the food-polygon area to the analyzed plate area.

Ocr Extraction Pipeline
Packaging images undergo four preprocessing stages before OCR: deskewing using the Hough Transform, deglare using Telea inpainting, CLAHE-based contrast normalization, and ROI cropping around the nutrition table. The processed region is then recognized using PaddleOCR PP-OCRv5. Layout reconstruction uses Y-axis grouping, while fuzzy matching based on edit distance maps OCR output to standardized nutrition terms. The choice of preprocessing is consistent with the broader observation that real-world food-label images are affected by packaging surfaces, illumination, layout variation, and recognition noise (Rosyadi et al., 2024; Saputra et al., 2024).

leva,bi,j=maxi,jifmini,j=0  (1)

Similaritya,b=1-leva,b/maxa,b (2)

Clinical Rules Engine
Nutrition values that have passed Human-in-the-Loop verification are evaluated by a deterministic rules engine. The system calculates Total Daily Energy Expenditure (TDEE) from Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equations. For diabetes-related dietary feedback, the implementation follows the locally specified PERKENI 2024 threshold for added/simple sugar and expresses the threshold relative to estimated daily energy requirements. For sodium-related feedback, the implementation is aligned with contemporary hypertension and diabetes nutrition guidance rather than relying exclusively on the older JNC 8 framework (“5. Facilitating Positive Health Behaviors and Well-Being to Improve Health Outcomes: Standards of Care in Diabetes—2024,” 2024; Kreutz et al., 2024). The resulting thresholds are used to classify accumulated intake into SAFE, WARNING, and DANGER states. These classifications are intended as transparent decision-support outputs rather than independent clinical diagnoses.
Male=10×W+6.25×H-5×A+5 (3)

Female=10×W+6.25×H-5×A-161 (4)
where BMR is the Basal Metabolic Rate (kcal/day), W is body weight (kg), H is height (cm), and A is age (years). For Type 2 Diabetes Mellitus, the daily pure-sugar limit follows PERKENI 2024 and is set dynamically at 5% of the user's total TDEE, rather than as a fixed value. For hypertension, the sodium limit follows the DASH Diet guideline (JNC 8) and is set as a static, absolute value of 2,000 mg per day, since the reference literature defines sodium thresholds as an absolute value rather than a proportion of caloric requirement. Based on these thresholds, the system classifies accumulated sugar or sodium intake into three risk levels: SAFE (well below the critical daily threshold), WARNING (reaching ≥75% of the daily threshold), and DANGER (significantly exceeding the threshold, or directly contradicting a user's medical contraindication).

Data Engineering
Data engineering is divided into two pipelines: packaging-image engineering for OCR and plate-image engineering for segmentation. In the OCR pipeline, incoming packaging images are checked for resolution, corrected for tilt (deskew), cleaned of reflections (deglare), and cropped to the nutrition-table region (ROI cropping); extracted text is filtered to remove non-nutritional words (e.g., promotional slogans, BPOM registration numbers) and mapped to standard nutrition terms via fuzzy string matching with a similarity threshold of ≥ 0.75. In the segmentation pipeline, the isipiringku_v3 dataset is enriched with spatial augmentation (random rotation, horizontal flip, scaling) and photometric augmentation (brightness, exposure, color-space conversion). Trained model weights are converted from Float32 to Float16 precision (best_float16.tflite), achieving roughly 50% file-size compression so they can be loaded quickly via TensorFlow.js WASM directly in the user's mobile browser (Offline Fallback Mode).
AI predictions may fail when an input falls outside the distribution represented in the training data. For this reason, Healthier includes a Segmentation Editor that allows users to correct predicted polygons and a Nutrient Value Verification Form that allows correction of OCR-extracted values before clinical rules are applied. This design follows the broader Human-in-the-Loop literature, which emphasizes human participation as a mechanism for handling uncertain predictions, improving data quality, and maintaining meaningful human control over machine-learning systems (Mosqueira-Rey et al., 2023).
To mitigate potential AI prediction errors  false positives in OCR or inaccurate object boundaries in segmentation Healthier provides two interactive features: a Segmentation Editor, an interactive canvas for visually correcting segmentation polygons before nutrition analysis, and a Nutrient Value Verification Form, allowing users to check and edit OCR-extracted nutrition figures before Clinical Rules Engine processing. Both serve as a human-validation layer that preserves system reliability when the AI model encounters conditions outside its generalization.
System usability was assessed using Nielsen's Heuristic Evaluation with four evaluators. Two evaluators independently reviewed the AI model, input variation, and generalization behavior; a third conducted an asynchronous architecture review; and a fourth checked implementation-documentation consistency. The fourth evaluator's findings were excluded from severity weighting, which was based on the independent E1-E2 assessment.
System usability was evaluated using the Nielsen Heuristic Evaluation method with four evaluators: two certified Machine Learning Engineers/Data Scientists as independent evaluators (E1, E2), one Lead Data Scientist conducting an asynchronous architecture review (E3), and one self-evaluator (E4) verifying documentation-implementation consistency. E4's findings were excluded from the severity-rating calculation to preserve objectivity, so severity weighting derives entirely from the E1–E2 consensus.
Table 2. Evaluator data
Code
Background
Evaluation Focus
Method
E1
Certified ML Engineer, applied-AI focus
AI model, input variation, generalization
Structured walkthrough
E2
Senior Data Scientist / AI Research Scientist
Architecture, HITL, OCR, segmentation
Structured walkthrough
E3
Lead Data Scientist / AI Engineer (Certified, USAII)
OCR vs. VLM, calorie estimation, code structure
Asynchronous architecture review
E4
System developer/researcher
Design and implementation consistency
Self-evaluation of documentation

RESULTS AND DISCUSSION

This section presents the results of implementation and empirical testing of the Healthier system: the quantitative evaluation of the two AI modules (segmentation localization and OCR text-extraction accuracy), the effectiveness of integrating both outputs through the Clinical Rules Engine, the role of the Human-in-the-Loop mechanism, and the results of Nielsen Heuristic Evaluation usability testing.

Instance Segmentation Evaluation
Evaluation on 3,529 validation images produced a Precision of 72.07%, Recall of 60.46%, and F1-Score of 65.45%. Mask mAP50 reached 67.06% and mAP50-95 reached 52.84%. The average inference time was 168 ms per image. These results are consistent with the broader literature showing that deep-learning-based food recognition and segmentation are technically feasible for dietary-assessment applications, although performance remains sensitive to food diversity, image conditions, and task definition (Dalakleidi et al., 2022; Telea, n.d.; Wahidin et al., 2024). Fruits achieved the highest class-level mAP50 at 81.44%, whereas the heterogeneous Others class achieved 53.37%. The lower Recall relative to Precision is important when interpreting downstream nutrition estimates and is therefore mitigated in Healthier through the interactive segmentation editor.

OCR Extraction Evaluation
The OCR module was evaluated on 30 Indonesian-market packaging photographs containing glossy, curved, and small-text packaging. Field-Level Accuracy was selected because the downstream system uses specific nutrition fields rather than unrestricted text transcription. Overall Field-Level Accuracy was 83.8%, with Calories/Energy reaching 93.3%, Protein 90.0%, and Serving Size 73.3%. The lower performance on Serving Size is consistent with the difficulty of extracting structured information from variable packaging layouts. Recent work on mobile nutrition assessment also demonstrates that data correction and user intervention can materially improve reliability when food databases, portion estimates, or packaged-food information are incomplete.

Clinical Rules Engine and Human-in-the-Loop Integration Results
The Clinical Rules Engine reproduced the TDEE calculation used by the implementation and generated deterministic SAFE, WARNING, and DANGER classifications from verified nutrition inputs. For example, a user weighing 70 kg, measuring 175 cm, and aged 25 years produces an estimated daily energy requirement of approximately 2,069 kcal under the implemented calculation. The system then evaluates sugar, carbohydrate, and sodium values against the configured thresholds. Recommendations are generated locally on the client, allowing the core rules to operate without continuous cloud inference. This design is consistent with the motivation for edge-oriented health applications, where local processing can reduce dependence on connectivity and centralized computation (Mosqueira-Rey et al., 2023; Ningrum & Mukminin, 2025). The HITL layer remains important because current evidence indicates that fully automated image-based dietary assessment still faces uncertainty in portion estimation and real-world food conditions.

Table 3. Summary of key findings
Aspect
Description
Key Findings
Segmentation reached Mask mAP50 67.06% (mAP50-95 52.84%) at 168 ms/image; OCR reached Field-Level Accuracy 83.8% (95% CI 78.6-87.9%); TDEE-based thresholds correctly triggered SAFE/WARNING/DANGER classification.
Comparison with Previous Research
Segmentation accuracy is broadly consistent with prior YOLO-based food studies, while surpassing them in guideline-based (Isi Piringku) category granularity. OCR accuracy is comparable to LLM-assisted pipelines, without the associated cloud cost and latency.
Phenomena and Explanations
Lower Recall than Precision in segmentation is attributed to visually heterogeneous classes (e.g., Others); lower OCR accuracy on Serving Size reflects inconsistent sentence-style formatting across brands rather than a model limitation.
Preliminary Conclusions
The combined pipeline is accurate and fast enough for real-time, edge-based use; Human-in-the-Loop correction is an effective safeguard for the remaining error margin in both modules.

The integrated evaluation indicates that Healthier's principal contribution is architectural rather than the claim that any individual AI component is state of the art. The segmentation module provides spatial food information, the OCR module extracts structured packaged-food data, the Clinical Rules Engine translates verified values into explicit thresholds, and HITL mechanisms allow users to intervene when the automated result is uncertain. This layered design reflects current evidence that image-based dietary assessment benefits from combining automated recognition with structured validation rather than treating model output as infallible (Chotwanvirat et al., 2024; Mosqueira-Rey et al., 2023; Shonkoff et al., 2023).
This section presents comprehensive testing results for Healthier, from the technical performance of both AI modules to their impact on user experience. Segmentation and OCR extraction were tested separately before being reviewed in an integrated manner through the Clinical Rules Engine, which translates raw data into health recommendations. The Human-in-the-Loop mechanism is discussed as a balance between automation and user control, particularly where the model encounters conditions outside its generalization limits. Finally, Nielsen Heuristic Evaluation results assess how well the interface design supports genuine user interaction.

CONCLUSIONS AND SUGGESTIONS

Conclusion
This research developed Healthier as an integrated nutrition-assistance system combining YOLO11s-seg food segmentation, PaddleOCR PP-OCRv5 label extraction, a deterministic Clinical Rules Engine, and Human-in-the-Loop verification. The segmentation model achieved a Mask mAP50 of 67.06% with an average inference time of 168 ms per image, while the OCR module achieved 83.8% Field-Level Accuracy. The results demonstrate the technical feasibility of combining visual food analysis and packaged-food label extraction in a local workflow. However, the findings should be interpreted within the study's validation scope: the OCR sample contains only 30 packaging photographs, the portion estimation remains based on 2D image information, and the Others class shows weaker segmentation performance. These limitations are consistent with challenges identified in recent systematic reviews of AI-based dietary assessment (Dalakleidi et al., 2022; Shonkoff et al., 2023).
The usability evaluation did not identify a design flaw that prevented core interaction, while the HITL components provided an explicit mechanism for correcting model output before nutrition rules were applied. Future work should prioritize broader validation on Indonesian foods, more rigorous portion or depth estimation, larger and more diverse packaging datasets, calibration of uncertainty, and prospective user studies. Lightweight deployment techniques such as quantization and further edge optimization are also appropriate directions, particularly given the resource constraints of mobile and edge devices (Yuan et al., 2024).
Future research should therefore focus on three areas: (1) expanding the Indonesian food and packaging datasets, including soups and visually complex foods; (2) improving portion estimation through depth, multi-angle imaging, or other geometric cues; and (3) evaluating the system prospectively with users and nutrition professionals. An offline Small Language Model may be explored as an optional component for explanation or normalization, but it should remain subordinate to deterministic nutrition rules and explicit human verification.
For future research, it is recommended to enrich the dataset with soup-based and texturally complex foods, apply model optimizations such as INT8 quantization, refine the Fuzzy Matching method in the OCR module, consider an offline Small Language Model (SLM), develop volume- or depth-based portion estimation through a multi-angle approach, and integrate the system with digital-health ecosystems such as Google Fit or Apple Health for more dynamic TDEE calculation.

REFERENCES

5. Facilitating Positive Health Behaviors and Well-being to Improve Health Outcomes: Standards of Care in Diabetes—2024. (2024). Diabetes Care, 47, S77–S110. https://doi.org/10.2337/dc24-S005
Aprianti Aprianti, Mubarokah Kismi, Yuantari MG Catur, & Rahma Nis SYifa’ur. (2023). Nutrition Fact Literacy in Productive Age Communities in Semarang City, Indonesia. 11(Nutrition Literacy).
Bonaccio, M., Di Castelnuovo, A., Ruggiero, E., Costanzo, S., Grosso, G., De Curtis, A., Cerletti, C., Donati, M. B., de Gaetano, G., & Iacoviello, L. (2023). Joint association of food nutritional profile by Nutri-Score front-of-pack label and ultra-processed food intake with mortality: Moli-sani prospective cohort study. BMJ, 378. https://doi.org/10.1136/bmj-2022-070688
Chotwanvirat, P., Prachansuwan, A., Sridonpai, P., & Kriengsinyos, W. (2024). Advancements in Using AI for Dietary Assessment Based on Food Images: Scoping Review. In Journal of Medical Internet Research (Vol. 26). JMIR Publications Inc. https://doi.org/10.2196/51432
Dalakleidi, K. V., Papadelli, M., Kapolos, I., & Papadimitriou, K. (2022). Applying Image-Based Food-Recognition Systems on Dietary Assessment: A Systematic Review. Advances in Nutrition, 13(6), 2590–2619. https://doi.org/10.1093/advances/nmac078
Folson, G. K., Bannerman, B., Atadze, V., Ador, G., Kolt, B., McCloskey, P., Gangupantulu, R., Arrieta, A., Braga, B. C., Arsenault, J., Kehs, A., Doyle, F., Tran, L. M., Hoang, N. T., Hughes, D., Nguyen, P. H., & Gelli, A. (2023). Validation of Mobile Artificial Intelligence Technology–Assisted Dietary Assessment Tool Against Weighed Records and 24-Hour Recall in Adolescent Females in Ghana. Journal of Nutrition, 153(8), 2328–2338. https://doi.org/10.1016/j.tjnut.2023.06.001
Ganderats-Fuentes, M., & Morgan, S. (2023). Front-of-Package Nutrition Labeling and Its Impact on Food Industry Practices: A Systematic Review of the Evidence. In Nutrients (Vol. 15, Number 11). Multidisciplinary Digital Publishing Institute (MDPI). https://doi.org/10.3390/nu15112630
Hananta, J. K., & Cahyono, N. (2026). Evaluation of YOLOv8 and Faster R-CNN for Image-Based Food Detection. In Journal of Applied Informatics and Computing (JAIC) (Vol. 10, Number 1). http://jurnal.polibatam.ac.id/index.php/JAIC
Kreutz, R., Brunström, M., Burnier, M., Grassi, G., Januszewicz, A., Muiesan, M. L., Tsioufis, K., de Pinho, R. M., Albini, F. L., Boivin, J. M., Doumas, M., Nemcsik, J., Rodilla, E., Agabiti-Rosei, E., Algharably, E. A. E., Agnelli, G., Benetos, A., Hitij, J. B., Cífková, R., … Mancia, G. (2024). 2024 European Society of Hypertension clinical practice guidelines for the management of arterial hypertension. European Journal of Internal Medicine, 126, 1–15. https://doi.org/10.1016/j.ejim.2024.05.033
Mohanty, S. P., Singhal, G., Scuccimarra, E. A., Kebaili, D., Héritier, H., Boulanger, V., & Salathé, M. (2022). The Food Recognition Benchmark: Using Deep Learning to Recognize Food in Images. Frontiers in Nutrition, 9. https://doi.org/10.3389/fnut.2022.875143
Mosqueira-Rey, E., Hernández-Pereira, E., Alonso-Ríos, D., Bobes-Bascarán, J., & Fernández-Leal, Á. (2023). Human-in-the-loop machine learning: a state of the art. Artificial Intelligence Review, 56(4), 3005–3054. https://doi.org/10.1007/s10462-022-10246-w
Moyen, A., Rappaport, A. I., Fleurent-Grégoire, C., Tessier, A. J., Brazeau, A. S., & Chevalier, S. (2022). Relative Validation of an Artificial Intelligence-Enhanced, Image-Assisted Mobile App for Dietary Assessment in Adults: Randomized Crossover Study. Journal of Medical Internet Research, 24(11). https://doi.org/10.2196/40449
Ningrum, V. S., & Mukminin, A. (2025). Efektivitas Program Isi Piringku terhadap Pemilihan Menu Cepat Saji di Posyandu Dahlia 1. Riwayat: Educational Journal of History and Humanities, 8(3), 3660–3674. https://doi.org/10.24815/jr.v8i3.48065
Oei, K., Choi, E. E. Y., Bar-Dayan, A., Stinson, J. N., Palmert, M. R., Alfonsi, J. E., & Hamilton, J. (2024). An Image-Recognition Dietary Assessment App for Adolescents With Obesity: Pilot Randomized Controlled Trial. JMIR Formative Research, 8. https://doi.org/10.2196/58682
Raju, V. B., Imtiaz, M. H., & Sazonov, E. (2023). Food Image Segmentation Using Multi-Modal Imaging Sensors with Color and Thermal Data. Sensors, 23(2). https://doi.org/10.3390/s23020560
Rosyadi, A. W., Ma’shumah, S., Zaman, M. Q., & Fajar, M. R. (2024). Ingredients Identification Through Label Scanning Using PaddleOCR and ChatGPT for Information Retrieval. Jurnal RESTI, 8(6), 758–767. https://doi.org/10.29207/resti.v8i6.6119
Saputra, T. M. E., Susanto, A., & Carmelita, B. J. (2024). Implementation of Tesseract OCR and Bounding Box for Text Extraction on Food Nutrition Labels. Building of Informatics, Technology and Science (BITS), 6(3), 1403–1412. https://doi.org/10.47065/bits.v6i3.6107
Shonkoff, E., Cara, K. C., Pei, X., Chung, M., Kamath, S., Panetta, K., & Hennessy, E. (2023). AI-based digital image dietary assessment methods compared to humans and ground truth: a systematic review. In Annals of Medicine (Vol. 55, Number 2). Taylor and Francis Ltd. https://doi.org/10.1080/07853890.2023.2273497
Shrestha, A., Cullerton, K., White, K. M., Mays, J., & Sendall, M. (2023). Impact of front-of-pack nutrition labelling in consumer understanding and use across socio-economic status: A systematic review. In Appetite (Vol. 187). Academic Press. https://doi.org/10.1016/j.appet.2023.106587
Telea, A. (n.d.). An Image Inpainting Technique Based on the Fast Marching Method (Vol. 9, Number 1).
Utami, G. C., Widiawati, C. R., & Subarkah, P. (2023). Detection of Indonesian Food to Estimate Nutritional Information Using YOLOv5. Teknika, 12(2), 158–165. https://doi.org/10.34148/teknika.v12i2.636
Wahidin, M., Achadi, A., Besral, B., Kosen, S., Nadjib, M., Nurwahyuni, A., Ronoatmodjo, S., Rahajeng, E., Pane, M., & Kusuma, D. (2024). Projection of diabetes morbidity and mortality till 2045 in Indonesia based on risk factors and NCD prevention and control programs. Scientific Reports, 14(1). https://doi.org/10.1038/s41598-024-54563-2
Wang, H., Tian, H., Ju, R., Ma, L., Yang, L., Chen, J., & Liu, F. (2024). Nutritional composition analysis in food images: an innovative Swin Transformer approach. Frontiers in Nutrition, 11. https://doi.org/10.3389/fnut.2024.1454466
Xiao, Z., Ling, R., & Deng, Z. (n.d.). FoodCSWin: A High-Accuracy Food Image Recognition Model for Dietary Assessment 2 learning; Diffusion model. Retrieved https://ssrn.com/abstract=4905090
Yuan, H., Kang, L., Li, Y., & Fan, Z. (2024). Human-in-the-loop machine learning for healthcare: Current progress and future opportunities in electronic health records. In Medicine Advances (Vol. 2, Number 3, pp. 318–322). John Wiley and Sons Inc. https://doi.org/10.1002/med4.70
Zhao, Y., Zhu, P., Jiang, Y., & Xia, K. (2024). Visual nutrition analysis: leveraging segmentation and regression for food nutrient estimation. Frontiers in Nutrition, 11. https://doi.org/10.3389/fnut.2024.1469878
Zhu, Z., & Dai, Y. (2023). A New CNN-Based Single-Ingredient Classification Model and Its Application in Food Image Segmentation. Journal of Imaging, 9(10). https://doi.org/10.3390/jimaging9100205



