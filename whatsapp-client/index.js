const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const { spawn } = require('child_process');
const path = require('path');

let flaskProcess = null;

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }
});

client.on('qr', (qr) => {
    console.log('\n=========================================');
    console.log('📱 WhatsApp Web Bağlantısı Bekleniyor');
    console.log('=========================================');
    console.log('Lütfen telefonunuzdan WhatsApp uygulamasını açın:');
    console.log('1. Ayarlar > Bağlı Cihazlar menüsüne girin.');
    console.log('2. "Cihaz Bağla" seçeneğine dokunun.');
    console.log('3. Aşağıdaki QR Kodu taratın.\n');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('\n✅ WhatsApp Bağlantısı Başarılı!');
    console.log('🤖 Bio-ML Köprüsü aktif. (Henüz Çekirdek Ajan başlatılmadı)');
    console.log('💬 Ajanı başlatmak için telefondan "STR" mesajını gönderin.');
});

client.on('message', async msg => {
    const text = msg.body.trim();
    if (!text) return;

    if (msg.from.includes('@g.us') || msg.from === 'status@broadcast') {
        return;
    }

    const upperText = text.toUpperCase();

    // 1. STR Komutu: Ajanı Başlat
    if (upperText === 'STR') {
        if (flaskProcess) {
            msg.reply('⚠️ Sistem zaten çalışıyor. Komut göndermek için "AGT" i ön ek olarak kullanın.');
            return;
        }

        msg.reply('⏳ Çekirdek ajan sunucusu başlatılıyor, lütfen bekleyin...');

        try {
            const scriptPath = path.resolve(__dirname, '../start_flask_only.sh');
            flaskProcess = spawn('bash', [scriptPath], { detached: true });

            flaskProcess.on('error', (err) => {
                console.error('Flask başlatılamadı:', err);
                msg.reply('❌ Ajan başlatılırken sistem hatası oluştu!');
                flaskProcess = null;
            });

            flaskProcess.on('exit', (code) => {
                console.log(`[İşlem] Flask sunucusu kapandı (Çıkış Kodu: ${code})`);
                flaskProcess = null;
            });

            // Başlatma marjı
            setTimeout(() => {
                msg.reply('✅ Ajan başarıyla başlatıldı ve servise hazır!\n\nArtık "AGT [komut]" formatında görev verebilirsiniz.\nÖrn: "AGT bana diyabet verisetini özetle."');
            }, 3000);
        } catch (e) {
            msg.reply('❌ Hata: ' + e.message);
            flaskProcess = null;
        }
        return;
    }

    // 2. AGT Filtresi: Sadece AGT ile başlayan komutları işletir
    if (!upperText.startsWith('AGT')) {
        return;
    }

    // Ajan kapalı ama komut gönderilmişse
    if (!flaskProcess) {
        msg.reply('❌ Sistem kapalı! Çekirdek ajanı uyandırmak için lütfen önce "STR" yazarak sistemi başlatın.');
        return;
    }

    // "AGT" kısmını komuttan ayıklama
    let cleanedText = text;
    if (upperText.startsWith('AGT ')) {
        cleanedText = text.substring(4).trim();
    } else {
        cleanedText = text.substring(3).trim();
    }

    console.log(`\n[WhatsApp] Ajan Görevlendirildi (${msg.from}): ${cleanedText}`);

    try {
        const response = await axios.post('http://127.0.0.1:5000/whatsapp-local', {
            text: cleanedText,
            from: msg.from
        });

        if (response.data && response.data.reply) {
            msg.reply(response.data.reply);
            console.log(`[WhatsApp] Yanıt iletildi.`);
        } else {
            msg.reply('Ajan bir yanıt üretemedi.');
        }
    } catch (error) {
        console.error('Flask API hatası:', error.message);
        msg.reply('❌ Çekirdek ajana ulaşılamadı. Python sunucusu çökmüş veya halen açılıyor olabilir. Lütfen biraz bekleyip tekrar deneyin veya kapatıp STR ile yeniden açın.');
        // Bağlantı koptuysa durumu temizle
        if (error.code === 'ECONNREFUSED') {
            flaskProcess = null;
        }
    }
});

// Ajan uygulamasını başlat
client.initialize();

// Sistemi güvenli kapatmak
process.on('SIGINT', () => {
    if (flaskProcess) {
        console.log('Çekirdek sunucu kapatılıyor...');
        try { process.kill(-flaskProcess.pid); } catch (e) { }
    }
    process.exit();
});
