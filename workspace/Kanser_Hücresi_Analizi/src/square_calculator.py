import csv
import os

def calculate_squares(start, end):
    """
    Belirtilen aralıktaki sayıların karelerini hesaplar.
    """
    squares = []
    for i in range(start, end + 1):
        squares.append({'Sayı': i, 'Karesi': i*i})
    return squares

def save_to_csv(data, filename):
    """
    Verilen veriyi bir CSV dosyasına kaydeder.
    """
    # ensure directory exists for the filename
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Sayı', 'Karesi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Kareler '{filename}' dosyasına başarıyla kaydedildi.")

if __name__ == "__main__":
    start_num = 1
    end_num = 5

    print(f"{start_num}'den {end_num}'e kadar olan sayıların kareleri hesaplanıyor...")
    square_data = calculate_squares(start_num, end_num)

    for item in square_data:
        print(f"Sayı: {item['Sayı']}, Karesi: {item['Karesi']}")

    # Çıktı dosyasını data/processed/ altına kaydedelim
    save_to_csv(square_data, 'data/processed/kareler.csv')