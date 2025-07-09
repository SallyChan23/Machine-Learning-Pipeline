# Submission 1: Prediksi Biaya Asuransi Kesehatan
Nama: Jeselyn Tania

Username dicoding: sallychan

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  |
| Masalah | Memprediksi besarnya biaya asuransi kesehatan yang harus dibayar berdasarkan data pribadi pengguna |
| Solusi machine learning | Menggunakan supervised learning dengan model regresi neural network (Keras) untuk prediksi biaya |
| Metode pengolahan | Data diproses melalui komponen TFX Transform untuk scaling dan encoding sesuai schema |
| Arsitektur model | Neural network sederhana: 1 input layer, 1 hidden layer (ReLU), 1 output layer (linear) |
| Metrik evaluasi | Mean Absolute Error (MAE) digunakan untuk menilai selisih rata-rata prediksi dan nilai asli |
| Performa model | MAE pada validation set sekitar 0.80 (nilai label sudah ditransformasi/di-scale sebelumnya) |
