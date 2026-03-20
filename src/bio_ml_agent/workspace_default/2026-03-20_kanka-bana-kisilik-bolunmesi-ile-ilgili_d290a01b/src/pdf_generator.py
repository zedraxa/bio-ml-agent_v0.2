import fpdf

class PDF(fpdf.FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 15)
        self.cell(0, 10, 'Dissosiyatif Kimlik Bozukluğu (DKB) Raporu', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('DejaVu', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def create_pdf_from_markdown(md_file_path, output_pdf_path):
    pdf = PDF()
    
    # Türkçe karakter desteği için DejaVu fontunu ekliyoruz
    # fpdf2'nin kendi font deposundan çekmesini sağlıyoruz
    try:
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        pdf.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
    except RuntimeError:
        print("Fontlar bulunamadı. Lütfen font dosyalarını doğru yola koyduğunuzdan emin olun.")
        # Alternatif olarak sistem fontlarını denemesini isteyebiliriz ama bu daha karmaşık.
        # Bu senaryoda genellikle font dosyaları paketle gelir.
        # Eğer yoksa, bir sonraki adımda wget ile indirilebilir.
        # Şimdilik devam edelim.
        pass

    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    with open(md_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            pdf.chapter_title(line[2:])
        elif line.startswith('## '):
            pdf.set_font('DejaVu', 'B', 11)
            pdf.cell(0, 8, line[3:], 0, 1, 'L')
            pdf.ln(2)
        elif line:
            pdf.chapter_body(line)
            
    pdf.output(output_pdf_path)
    print(f"PDF dosyası başarıyla oluşturuldu: {output_pdf_path}")

if __name__ == "__main__":
    # DejaVu font dosyalarını indirelim (eğer yoksa)
    import os
    import urllib.request
    
    FONT_DIR = os.path.join(os.path.dirname(fpdf.__file__), 'font')
    DEJAVU_SANS_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
    DEJAVU_BOLD_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf"
    DEJAVU_ITALIC_URL = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Oblique.ttf"
    
    def download_font(url, dest_path):
        if not os.path.exists(dest_path):
            print(f"Font indiriliyor: {os.path.basename(dest_path)}")
            urllib.request.urlretrieve(url, dest_path)

    download_font(DEJAVU_SANS_URL, os.path.join(FONT_DIR, 'DejaVuSans.ttf'))
    download_font(DEJAVU_BOLD_URL, os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf'))
    download_font(DEJAVU_ITALIC_URL, os.path.join(FONT_DIR, 'DejaVuSans-Oblique.ttf'))

    create_pdf_from_markdown('rapor_icerik.md', 'Kişilik_Bolunmesi_Raporu.pdf')

---