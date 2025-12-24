import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Fungsi keanggotaan untuk permukaan lantai (halus, normal, kasar) 1-100
permukaan_halus = np.linspace(0, 100, 101)
permukaan_normal = np.linspace(0, 100, 101)
permukaan_kasar = np.linspace(0, 100, 101)

# Fungsi keanggotaan menggunakan trimf (segitiga)
membership_permukaan_halus = fuzz.trimf(permukaan_halus, [0, 30, 50])  # Memodifikasi titik-titik keanggotaan
membership_permukaan_normal = fuzz.trimf(permukaan_normal, [30, 50, 70])
membership_permukaan_kasar = fuzz.trimf(permukaan_kasar, [50, 70, 100])

# Fungsi keanggotaan untuk jumlah debu (sedikit, normal, banyak) 1-100
debu_sedikit = np.linspace(0, 100, 101)
debu_normal = np.linspace(0, 100, 101)
debu_banyak = np.linspace(0, 100, 101)

membership_debu_sedikit = fuzz.trimf(debu_sedikit, [0, 30, 50])
membership_debu_normal = fuzz.trimf(debu_normal, [30, 50, 70])  # Keanggotaan debu normal
membership_debu_banyak = fuzz.trimf(debu_banyak, [50, 70, 100])

# Fungsi untuk menghitung derajat keanggotaan berdasarkan input
def hitung_derajat_keanggotaan_permukaan(nilai_permukaan):
    keanggotaan_halus = fuzz.interp_membership(permukaan_halus, membership_permukaan_halus, nilai_permukaan)
    keanggotaan_normal = fuzz.interp_membership(permukaan_normal, membership_permukaan_normal, nilai_permukaan)
    keanggotaan_kasar = fuzz.interp_membership(permukaan_kasar, membership_permukaan_kasar, nilai_permukaan)
    return keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar

def hitung_derajat_keanggotaan_debu(nilai_debu):
    keanggotaan_sedikit = fuzz.interp_membership(debu_sedikit, membership_debu_sedikit, nilai_debu)
    keanggotaan_normal = fuzz.interp_membership(debu_normal, membership_debu_normal, nilai_debu)
    keanggotaan_banyak = fuzz.interp_membership(debu_banyak, membership_debu_banyak, nilai_debu)
    return keanggotaan_sedikit, keanggotaan_normal, keanggotaan_banyak

# Fungsi untuk menghitung Z (dengan rumus perhitungan KP dan Z)
def hitung_Z(alpha, Z_min, Z_max):
    Z = Z_max - (alpha * (Z_max - Z_min))
    return Z

# Inferensi Fuzzy berdasarkan aturan yang diberikan
def inferensi_fuzzy(keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar, keanggotaan_sedikit, keanggotaan_normal_debu, keanggotaan_banyak):
    # Aturan 1: Lantai Halus, Debu Sedikit -> Kecepatan Pelan
    alpha_R1 = min(keanggotaan_halus, keanggotaan_sedikit)
    Z1 = hitung_Z(alpha_R1, 30, 50)  # Rumus perhitungan Z1 (50-Z1)/(50-30)

    # Aturan 2: Lantai Halus, Debu Normal -> Kecepatan Normal
    alpha_R2 = min(keanggotaan_halus, keanggotaan_normal_debu)
    Z2 = hitung_Z(alpha_R2, 30, 50)  # Rumus perhitungan Z2 (50-Z2)/(50-30)

    # Aturan 3: Lantai Halus, Debu Banyak -> Kecepatan Normal
    alpha_R3 = min(keanggotaan_halus, keanggotaan_banyak)
    Z3 = hitung_Z(alpha_R3, 30, 50)  # Rumus perhitungan Z3 (50-Z3)/(50-30)

    # Aturan 4: Lantai Normal, Debu Sedikit -> Kecepatan Normal
    alpha_R4 = min(keanggotaan_normal, keanggotaan_sedikit)
    Z4 = hitung_Z(alpha_R4, 50, 30)  # Rumus perhitungan Z4 (Z4-30)/(50-30)

    # Aturan 5: Lantai Normal, Debu Normal -> Kecepatan Normal
    alpha_R5 = min(keanggotaan_normal, keanggotaan_normal_debu)
    Z5 = hitung_Z(alpha_R5, 50, 30)  # Rumus perhitungan Z5 (Z5-30)/(50-30)

    # Aturan 6: Lantai Normal, Debu Banyak -> Kecepatan Cepat
    alpha_R6 = min(keanggotaan_normal, keanggotaan_banyak)
    Z6 = hitung_Z(alpha_R6, 50, 30)  # Rumus perhitungan Z6 (Z6-30)/(50-30)

    # Aturan 7: Lantai Kasar, Debu Sedikit -> Kecepatan Normal
    alpha_R7 = min(keanggotaan_kasar, keanggotaan_sedikit)
    Z7 = hitung_Z(alpha_R7, 70, 50)  # Rumus perhitungan Z7 (Z7-50)/(70-50)

    # Aturan 8: Lantai Kasar, Debu Normal -> Kecepatan Cepat
    alpha_R8 = min(keanggotaan_kasar, keanggotaan_normal_debu)
    Z8 = hitung_Z(alpha_R8, 70, 50)  # Rumus perhitungan Z8 (Z8-50)/(70-50)

    # Aturan 9: Lantai Kasar, Debu Banyak -> Kecepatan Cepat
    alpha_R9 = min(keanggotaan_kasar, keanggotaan_banyak)
    Z9 = hitung_Z(alpha_R9, 70, 50)  # Rumus perhitungan Z9 (Z9-50)/(70-50)

    # Mengembalikan semua nilai α dan Z
    return [alpha_R1, alpha_R2, alpha_R3, alpha_R4, alpha_R5, alpha_R6, alpha_R7, alpha_R8, alpha_R9], \
           [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9]

# Fungsi defuzzifikasi
def defuzzifikasi(alphas, Zs):
    num = sum(alpha * Z for alpha, Z in zip(alphas, Zs))
    denom = sum(alphas)
    return num / denom if denom != 0 else 0

# Fungsi untuk visualisasi keanggotaan
def plot_keanggotaan(nilai_permukaan, nilai_debu, keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar, keanggotaan_sedikit, keanggotaan_normal_debu, keanggotaan_banyak):
    # Plot Permukaan Lantai
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(permukaan_halus, membership_permukaan_halus, label="Lantai Halus")
    plt.plot(permukaan_normal, membership_permukaan_normal, label="Lantai Normal")
    plt.plot(permukaan_kasar, membership_permukaan_kasar, label="Lantai Kasar")
    plt.scatter(nilai_permukaan, keanggotaan_halus, color='red', zorder=5)
    plt.scatter(nilai_permukaan, keanggotaan_normal, color='red', zorder=5)
    plt.scatter(nilai_permukaan, keanggotaan_kasar, color='red', zorder=5)
    plt.title("Keanggotaan Permukaan Lantai")
    plt.xlabel("Persentase Permukaan Lantai")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()

    # Plot Jumlah Debu
    plt.subplot(2, 2, 2)
    plt.plot(debu_sedikit, membership_debu_sedikit, label="Debu Sedikit")
    plt.plot(debu_normal, membership_debu_normal, label="Debu Normal")
    plt.plot(debu_banyak, membership_debu_banyak, label="Debu Banyak")
    plt.scatter(nilai_debu, keanggotaan_sedikit, color='red', zorder=5)
    plt.scatter(nilai_debu, keanggotaan_normal_debu, color='red', zorder=5)
    plt.scatter(nilai_debu, keanggotaan_banyak, color='red', zorder=5)
    plt.title("Keanggotaan Jumlah Debu")
    plt.xlabel("Persentase Jumlah Debu")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Fungsi utama untuk menerima input dari pengguna
def main():
    print("Program ini akan membantu menentukan kecepatan daya hisap vacuum cleaner berdasarkan permukaan lantai dan jumlah debu.")

    # Penjelasan untuk memilih nilai permukaan lantai
    print("\nMasukkan nilai untuk permukaan lantai (1 sampai 100)%")
    try:
        nilai_permukaan = float(input("Nilai Permukaan Lantai: "))
        if nilai_permukaan < 1 or nilai_permukaan > 100:
            print("Nilai permukaan lantai harus di antara 1 dan 100.")
            return
    except ValueError:
        print("Masukkan nilai numerik yang valid untuk permukaan lantai.")
        return

    # Penjelasan untuk memilih nilai debu
    print("\nMasukkan nilai untuk jumlah debu yang terdeteksi (1 sampai 100)%")
    try:
        nilai_debu = float(input("Nilai Jumlah Debu: "))
        if nilai_debu < 1 or nilai_debu > 100:
            print("Nilai debu harus di antara 1 dan 100.")
            return
    except ValueError:
        print("Masukkan nilai numerik yang valid untuk debu.")
        return

    # Hitung derajat keanggotaan
    keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar = hitung_derajat_keanggotaan_permukaan(nilai_permukaan)
    keanggotaan_sedikit, keanggotaan_normal_debu, keanggotaan_banyak = hitung_derajat_keanggotaan_debu(nilai_debu)

    # Tampilkan hasil fuzzifikasi
    print("\nHasil Fuzzifikasi untuk Permukaan Lantai:")
    print(f"Keanggotaan Lantai Halus   : {keanggotaan_halus:.2f}")
    print(f"Keanggotaan Lantai Normal  : {keanggotaan_normal:.2f}")
    print(f"Keanggotaan Lantai Kasar   : {keanggotaan_kasar:.2f}")

    print("\nHasil Fuzzifikasi untuk Jumlah Debu:")
    print(f"Keanggotaan Debu Sedikit   : {keanggotaan_sedikit:.2f}")
    print(f"Keanggotaan Debu Normal    : {keanggotaan_normal_debu:.2f}")
    print(f"Keanggotaan Debu Banyak    : {keanggotaan_banyak:.2f}")

    # Inferensi dan defuzzifikasi
    alphas, Zs = inferensi_fuzzy(keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar, keanggotaan_sedikit, keanggotaan_normal_debu, keanggotaan_banyak)

    # Menampilkan nilai alpha dan Z
    print("\nNilai Alpha dan Z (Setiap Aturan):")
    for i, (alpha, Z) in enumerate(zip(alphas, Zs), 1):
        print(f"Aturan {i}: Alpha = {alpha:.2f}, Z = {Z:.2f}")

    hasil_defuzzifikasi = defuzzifikasi(alphas, Zs)

    # Menampilkan hasil defuzzifikasi dengan koma (bernilai desimal)
    print("\nHasil Defuzzifikasi (Kecepatan Daya Hisap) sebelum dibulatkan:")
    print(f"Kecepatan Daya Hisap: {hasil_defuzzifikasi:.2f}")

    # Pembulatan hasil defuzzifikasi
    hasil_bulat = round(hasil_defuzzifikasi)

    # Menampilkan hasil setelah dibulatkan
    print(f"Kecepatan Daya Hisap (Output): {hasil_bulat:.0f}")

    # Menambahkan kesimpulan
    if hasil_bulat <= 40:
        print("Kesimpulan: Kecepatan Daya Hisap = Pelan")
    elif 41 <= hasil_bulat <= 60:
        print("Kesimpulan: Kecepatan Daya Hisap = Normal")
    else:
        print("Kesimpulan: Kecepatan Daya Hisap = Cepat")

    # Visualisasi keanggotaan
    plot_keanggotaan(nilai_permukaan, nilai_debu, keanggotaan_halus, keanggotaan_normal, keanggotaan_kasar, keanggotaan_sedikit, keanggotaan_normal_debu, keanggotaan_banyak)

# Panggil fungsi utama
if __name__ == "__main__":
    main()
