# TA-03 Credit Risk Prediction

Proyek pembelajaran mesin untuk memprediksi risiko gagal bayar nasabah kartu kredit menggunakan Logistic Regression dan Logistic Regression dengan SMOTENC. Proyek mencakup eksplorasi data, prapemrosesan, evaluasi model, serta integrasi model ke aplikasi web menggunakan Flask.

## Tujuan

- Membangun model klasifikasi untuk memprediksi gagal bayar kartu kredit.
- Membandingkan Logistic Regression tanpa penyeimbangan kelas dan Logistic Regression dengan SMOTENC.
- Mengevaluasi model menggunakan Accuracy, Precision, Recall, F1-score, ROC-AUC, dan Confusion Matrix.
- Menampilkan prediksi risiko kredit melalui aplikasi web Flask.

## Dataset

Dataset yang digunakan adalah **Default of Credit Card Clients** dari UCI Machine Learning Repository.

Sumber: https://doi.org/10.24432/C55S3H

Dataset awal memiliki 30.000 baris dan 25 kolom. Setelah seleksi fitur dan pembersihan, 29.856 baris digunakan untuk pemodelan.

Variabel target:
- `0`: Tidak gagal bayar.
- `1`: Gagal bayar pada bulan berikutnya.

Sembilan fitur prediktor yang digunakan:
- `LIMIT_BAL`
- `AGE`
- `EDUCATION`
- `MARRIAGE`
- `PAY_0`
- `PAY_2`
- `PAY_3`
- `BILL_AMT1`
- `PAY_AMT1`

**Catatan:** File dataset tidak disertakan dalam repositori apabila dikecualikan oleh `.gitignore`. Unduh dataset dari sumber aslinya dan letakkan pada folder `data/` sebelum menjalankan notebook atau pelatihan ulang.

## Teknologi

- Python
- Pandas dan NumPy
- Scikit-learn
- imbalanced-learn
- Joblib
- Flask
- HTML, CSS, dan JavaScript

## Struktur Proyek

```text
TA-03-Credit-Risk/
├── data/
│   └── UCI_Credit_Card.csv
├── notebooks/
│   └── TA03_Credit_Risk_Analysis.ipynb
├── models/
│   ├── credit_risk_model.joblib
│   ├── model_metadata.json
│   ├── model_comparison.csv
│   └── model_coefficients.csv
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
├── vercel.json
└── README.md
```

Nama file di atas menggambarkan struktur proyek yang digunakan. Pastikan nama file pada repositori sesuai dengan file yang benar-benar tersedia.

## Hasil Evaluasi

Evaluasi menggunakan 5.972 data testing.

| Metrik | Logistic Regression | Logistic Regression + SMOTENC |
|---|---:|---:|
| Accuracy | 0,8069 | 0,6490 |
| Precision Default | 0,7007 | 0,3427 |
| Recall Default | 0,2231 | 0,6377 |
| F1-score Default | 0,3385 | 0,4458 |
| ROC-AUC | 0,7072 | 0,6990 |

Model dengan SMOTENC memperoleh recall dan F1-score kelas Default yang lebih tinggi. Recall meningkat dari 0,2231 menjadi 0,6377. Artinya, model dapat mendeteksi lebih banyak kasus gagal bayar pada data pengujian.

Namun, accuracy dan precision menurun, sedangkan false positive meningkat. Oleh karena itu, model dengan SMOTENC dipilih untuk demonstrasi ini berdasarkan prioritas deteksi gagal bayar, bukan karena unggul pada semua metrik.

## Menjalankan Aplikasi Secara Lokal

Pastikan Python telah terpasang, kemudian buka terminal pada folder utama proyek.

### 1. Buat virtual environment

```bash
python -m venv .venv
```

### 2. Aktifkan virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika aktivasi diblokir oleh kebijakan PowerShell, gunakan terminal Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Instal dependensi

```bash
python -m pip install -r requirements.txt
```

### 4. Periksa file model

Pastikan file model tersedia di lokasi yang diharapkan aplikasi, yaitu `models/credit_risk_model.joblib`.

Jika file model belum tersedia, jalankan notebook atau skrip pelatihan yang sesuai untuk menghasilkan model tersebut. Pelatihan ulang dapat menghasilkan model yang berbeda jika konfigurasi, data, atau versi library berubah.

### 5. Jalankan Flask

```bash
python app.py
```

Buka alamat berikut pada browser:

http://127.0.0.1:5000

Jika aplikasi menyediakan endpoint pemeriksaan kesehatan, endpoint tersebut dapat diuji melalui:

http://127.0.0.1:5000/health

## Deployment

Aplikasi dapat di-deploy ke platform cloud yang mendukung aplikasi Python/Flask, sesuai konfigurasi proyek.

- **URL aplikasi:** [https://ai-3-ml.vercel.app/]
- **Repositori GitHub:** [https://github.com/ReidOates/AI3ML]

Setelah deployment, lakukan pengujian pada halaman utama, pengiriman formulir, pemuatan model, dan tampilan hasil prediksi. Jangan menganggap aplikasi berhasil di-deploy sebelum pengujian tersebut selesai.

## Batasan

- Dataset berasal dari data historis nasabah kartu kredit di Taiwan dan tidak otomatis mewakili kondisi nasabah saat ini.
- Hasil model merupakan prediksi statistik, bukan kepastian bahwa nasabah akan gagal bayar.
- Model dengan SMOTENC meningkatkan recall, tetapi menghasilkan lebih banyak false positive.
- Aplikasi ini merupakan demonstrasi akademik dan bukan sistem keputusan kredit siap operasional.

## Referensi

- Yeh, I. C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020
- imbalanced-learn developers. *SMOTENC documentation*. https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html
- scikit-learn developers. *Model evaluation documentation*. https://scikit-learn.org/stable/modules/model_evaluation.html
