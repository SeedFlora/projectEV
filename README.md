## ICDEES 2026 revised-paper artifacts

The reproducibility materials for **Shrinkage-Adaptive Weighted Soft Voting for Ensembles of Indonesian Pre-Trained Language Models in Electric Vehicle Sentiment Classification** are now included: numerical labels from both labelling passes, experiment code, five-seed stored posteriors/fold indices, and aggregate results (primary text/worksheets are withheld pending privacy and redistribution review). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for provenance, external-corpus source links and reproduction commands. The legacy prototype and single-split table below are retained as project history; they are not the revised paper's multi-seed benchmark table.

# projectEV - Analisis Sentimen EV untuk Insight Bisnis

## Ringkas
Project ini membangun pipeline analisis sentimen EV berbahasa Indonesia, mengevaluasi 5 model Transformer, menampilkan dashboard analitik (distribusi, top kata, word cloud), dan menyediakan batch processing serta deploy ke Hugging Face + Gradio.

## Latar Belakang
Persepsi publik terhadap EV di Indonesia terlihat jelas di kanal media sosial. Studi sentiment berbasis Twitter dan media sosial di Indonesia menunjukkan variasi emosi publik terhadap EV, faktor harga, dan isu infrastruktur yang berulang muncul sebagai topik penting. Temuan ini menegaskan kebutuhan analisis sentimen yang terukur untuk mendukung pengambilan keputusan bisnis (Salsabila et al., 2023; Michael & Fianty, 2024; Seraphina & Gunawan, 2025).

Selain konteks lokal, studi berbasis Twitter di wilayah rural AS memperlihatkan bahwa sentimen terkait EV dan infrastruktur berbeda antar wilayah dan dipengaruhi ketersediaan charging serta isu keselamatan. Temuan di tingkat wilayah ini menunjukkan bahwa brand perlu memahami variasi persepsi berdasarkan konteks geografis dan kesiapan infrastruktur (Wang et al., 2024).

Dari sisi adopsi, riset perilaku konsumen menekankan bahwa niat beli EV dipengaruhi persepsi manfaat, kemudahan, dan risiko, serta norma sosial. Studi di China dengan pendekatan TAM/TPB dan perluasan variabel menunjukkan adanya ambivalensi sikap, nilai yang dirasakan, dan faktor psikologis sebagai penentu niat beli (Zhang et al., 2023; Hu et al., 2025; Yegin & Ikram, 2022; Handarujati, 2024).

Infrastruktur charging menjadi faktor kritikal dalam adopsi dan kepuasan. Preferensi lokasi charging (mis. aksesibilitas, demografi) serta isu reliability charger publik terbukti memengaruhi niat beli EV. Hal ini menunjukkan pentingnya strategi jaringan charging yang andal dan terencana (Bhat et al., 2024; Singh et al., 2025).

Riset teknis di sisi perencanaan dan evaluasi charging menekankan kebutuhan metode data-driven dan metrik kualitas layanan (QoE) untuk memproyeksi beban dan mengoptimalkan ekspansi. Studi perencanaan multi-periode dan review model distribusi charging menyarankan bahwa keputusan investasi perlu berbasis data lalu lintas dan perilaku pengemudi (Al-Dahabreh et al., 2023; He et al., 2023; Lin et al., 2024).

Pada domain baterai, dataset publik skala besar dan studi diagnosis/prognosis menunjukkan bahwa kesehatan baterai, biaya perawatan, dan estimasi umur pakai merupakan isu bisnis utama. Ini menguatkan perlunya insight yang menyorot faktor baterai dan biaya operasional (He et al., 2022; Zhao & Burke, 2022).

## Rumusan Masalah
1. Bagaimana memetakan sentimen publik EV secara akurat dari teks berbahasa Indonesia?
2. Faktor apa yang paling sering muncul dan relevan untuk keputusan bisnis (harga, baterai, charging, layanan)?
3. Bagaimana menyajikan insight agar mudah dipahami oleh pengambil keputusan non-teknis?
4. Model Transformer mana yang paling stabil untuk domain ini?

## Tujuan
1. Melakukan klasifikasi sentimen (positif/negatif) pada teks EV berbahasa Indonesia.
2. Mengevaluasi 5 model Transformer dan memilih yang terbaik secara F1.
3. Menyediakan dashboard analitik dengan visualisasi dan insight bisnis.
4. Menyediakan batch processing dan deploy ke Hugging Face + Gradio.

## Manfaat
1. Mengidentifikasi isu bisnis paling dominan (harga, baterai, charging, layanan purna jual).
2. Memberi sinyal dini terhadap risiko reputasi dan masalah produk.
3. Membantu prioritas pengembangan fitur dan strategi komunikasi.
4. Mendukung keputusan ekspansi jaringan charging dan pricing.

## Metodologi Singkat
- Data: komentar/teks EV berbahasa Indonesia, dilabel ulang dan diseimbangkan (positif/negatif).
- Split: train/val/test.
- Model: 5 Transformer (IndoBERT, BERT Indonesian, RoBERTa Indonesian, XLM-R, mBERT).
- Metrik: accuracy, precision, recall, F1.
- Analitik: top kata dan word cloud difokuskan ke kata bermakna bisnis.

## Evaluasi 5 Model (Ringkas)

| Model | Val F1 | Test F1 |
| --- | --- | --- |
| indobenchmark/indobert-base-p1 | 0.899 | 0.906 |
| cahya/bert-base-indonesian-1.5G | 0.912 | 0.906 |
| cahya/roberta-base-indonesian-1.5G | 0.887 | 0.892 |
| xlm-roberta-base | 0.846 | 0.828 |
| bert-base-multilingual-cased | 0.828 | 0.766 |

Model terbaik pada eksperimen ini adalah **cahya/bert-base-indonesian-1.5G** berdasarkan val_f1 dan stabil di test_f1.

## Deployment
- Model HF: https://huggingface.co/seedflora/ev-sentiment
- Space Gradio: https://huggingface.co/spaces/seedflora/ev-sentiment-dashboard

Fitur Space:
- Prediksi tunggal.
- Batch processing (upload CSV/XLSX).
- Dashboard analitik: distribusi label, top kata, word cloud, dan ringkasan insight.

## Daftar Pustaka (>=20 paper, 2021-2026)
1. Salsabila, H., Habibi, R., & Harani, N. H. (2023). Social Media-Based Sentiment Analysis: Electric Vehicle Usage in Indonesia. The Indonesian Journal of Computer Science. https://ijcs.net/ijcs/index.php/ijcs/article/view/3250
2. Michael, M., & Fianty, M. I. (2024). Measuring Public Sentiment on Electric Cars in Indonesia: A Social Media Data Analysis. G-Tech: Jurnal Teknologi Terapan. https://ejournal.uniramalang.ac.id/g-tech/article/view/5405
3. Wang, M. B., Zhao, L., & Cochran, A. L. (2024). Sentiments of Rural U.S. Communities on Electric Vehicles and Infrastructure: Insights from Twitter Data. Sustainability. https://www.mdpi.com/2071-1050/16/11/4871
4. Chairat, A. S. N., Rizal, R., & Himawan, H. (2026). Comparative Sentiment Analysis of YouTube Comments on Indonesia's Electric Vehicle Incentive Policy Using TF-IDF and IndoBERTweet Models. JUTIF. https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/5499
5. Seraphina, Y. A., & Gunawan, P. H. (2025). Sentiment Analysis of Public Responses Regarding The Use of Electric Cars in Indonesia with SVM and Random Forest. The Indonesian Journal of Computer Science. https://ijcs.net/ijcs/index.php/ijcs/article/view/4649
6. Sitorus, Z., Saputra, M., Sofyan, S. N., & Susilawati. (2024). Sentiment Analysis of Indonesian Community Towards Electric Motorcycles on Twitter Using Orange Data Mining. INFOTECH Journal. https://www.ejournal.unma.ac.id/index.php/infotech/article/view/9374
7. Zhang, L., Tong, H., Liang, Y., & Qin, Q. (2023). Consumer purchase intention of new energy vehicles with an extended TAM. Transportation Research Part A. https://econpapers.repec.org/RePEc:eee:transa:v:174:y:2023:i:c:s0965856423001623
8. Hu, X., Yusof, R. N. R., & Mansor, Z. D. (2025). Consumers' Purchase Intentions Towards NEVs Based on TPB on Perceived Value. World Electric Vehicle Journal. https://www.mdpi.com/2032-6653/16/3/120
9. Handarujati, E. (2024). Analysis of Factors Driving Purchase Intention of Electric Cars: TPB, NAM, TAM. Journal of Economics and Business UBS. https://www.jurnal.ubs-usg.ac.id/index.php/joeb/article/view/1279
10. Yegin, T., & Ikram, M. (2022). Analysis of Consumers' EV Purchase Intentions: Expansion of TPB. Sustainability. https://ideas.repec.org/a/gam/jsusta/v14y2022i19p12091-d924126.html
11. Bhat, F. A., Tiwari, G. Y., & Verma, A. (2024). Preferences for public EV charging infrastructure locations: A discrete choice analysis. Transport Policy. https://trid.trb.org/View/2340101
12. Singh, R., Quinn, C., & MacKenzie, D. (2025). Poor reliability of public charging stations can impede the growth of the EV market. Transport Policy. https://www.osti.gov/pages/biblio/2575447-poor-reliability-public-charging-stations-can-impede-growth-electric-vehicle-market
13. Lin, W., Wei, H., Yang, L., & Zhao, X. (2024). Technical review of EV charging distribution models with driver behavior impacts. JTTE. https://jtte.chd.edu.cn/article/doi/10.1016/j.jtte.2024.06.001
14. Al-Dahabreh, N., Sayed, M. A., Sarieddine, K., Elhattab, M., Khabbaz, M., Atallah, R., & Assi, C. (2023). A Data-Driven Framework for Improving Public EV Charging Infrastructure: Modeling and Forecasting. arXiv. https://arxiv.org/abs/2312.05333
15. He, M., Krishnakumari, P., Luo, D., & Chen, J. (2023). A Data-Driven Integrated Framework for Fast-Charging Facility Planning using Multi-Period Bi-Objective Optimization. arXiv. https://arxiv.org/abs/2311.12700
16. Mehditabrizi, A., Tahmasbi, B., Namadi, S. S., & Cirillo, C. (2024). Integrating En Route and Home Proximity in EV Charging Accessibility. arXiv. https://arxiv.org/abs/2409.08287
17. Aduama, P., Zhang, Z., & Al Sumaiti, A. S. (2023). Multi Feature Data Fusion-Based Load Forecasting of EV Charging Stations Using a Deep Learning Model. arXiv. https://arxiv.org/abs/2301.13774
18. Alshehhi, B., Karapetyan, A., Elbassioni, K., Chau, S. C.-K., & Khonji, M. (2023). DClEVerNet: Deep Combinatorial Learning for Efficient EV Charging Scheduling. arXiv. https://arxiv.org/abs/2305.11195
19. He, H., Zhang, J., Wang, Y., Jiang, B., Huang, S., Wang, C., Zhang, Y., Xiong, G., Han, X., Guo, D., He, G., & Ouyang, M. (2022). EVBattery: A Large-Scale EV Dataset for Battery Health and Capacity Estimation. arXiv. https://arxiv.org/abs/2201.12358
20. Zhao, J., & Burke, A. F. (2022). Electric Vehicle Batteries: Status and Perspectives of Data-Driven Diagnosis and Prognosis. Batteries. https://www.mdpi.com/2313-0105/8/10/142
