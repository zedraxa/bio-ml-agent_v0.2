#!/bin/bash
# Bio-ML Agent WhatsApp Bot Başlatıcı (Yerel / QR Kodlu)

echo "====================================================="
echo "📱 Bio-ML WhatsApp Botu Başlatılıyor..."
echo "====================================================="
echo "Adımlar:"
echo "1. Aşağıdaki QR kodunu telefonunuzun WhatsApp 'Bağlı Cihazlar' kısmından okutunuz."
echo "2. Bağlantı kurulduğunda WhatsApp'tan 'STR' yazarak çekirdek ajanı başlatın."
echo "3. Ajan açıldıktan sonra 'AGT mesajiniz' formatıyla komut gönderin."
echo "====================================================="

# whatsapp-client dizinine geç ve Node.js uygulamasını ön planda başlat
cd whatsapp-client
node index.js
