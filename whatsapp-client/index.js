const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcodeTerminal = require('qrcode-terminal');
const QRCode = require('qrcode');
const fs = require('fs');
const axios = require('axios');
const { spawn } = require('child_process');
const path = require('path');
const express = require('express');
const yaml = require('js-yaml');

let flaskProcess = null;
let currentState = 'INIT';
let lastQr = null;
const FLASK_HEALTH_URL = 'http://127.0.0.1:5000/health';
const ALLOW_NO_SANDBOX = process.env.WHATSAPP_ALLOW_NO_SANDBOX === '1';
const API_KEY = (() => {
    try {
        const configPath = path.resolve(__dirname, '../config.yaml');
        const parsed = yaml.load(fs.readFileSync(configPath, 'utf8'));
        return parsed?.security?.api_key || '';
    } catch {
        return '';
    }
})();

const chromiumArgs = [
    '--disable-dev-shm-usage',
    '--disable-accelerated-2d-canvas',
    '--no-first-run',
    '--no-zygote',
    '--single-process',
    '--disable-gpu',
    '--ash-no-nudges',
    '--disable-background-networking',
    '--disable-background-timer-throttling',
    '--disable-client-side-phishing-detection',
    '--disable-default-apps',
    '--disable-extensions',
    '--disable-hang-monitor',
    '--disable-prompt-on-repost',
    '--disable-sync',
    '--disable-translate',
    '--metrics-recording-only',
    '--mute-audio',
    '--password-store=basic',
    '--use-mock-keychain',
    '--disable-blink-features=AutomationControlled'
];

if (ALLOW_NO_SANDBOX) {
    chromiumArgs.unshift('--disable-setuid-sandbox');
    chromiumArgs.unshift('--no-sandbox');
}

async function isFlaskAlive() {
    try {
        const resp = await axios.get(FLASK_HEALTH_URL, { timeout: 2000 });
        return resp.status === 200;
    } catch {
        return false;
    }
}

// ─────────────────────────────────────────────
//  ToS Compliance Check (Faz 1)
// ─────────────────────────────────────────────
if (!process.argv.includes('--accept-tos')) {
    console.error('\n======================================================');
    console.error(' ❌ DİKKAT: WhatsApp ToS İhlali Riski (UYARI) ');
    console.error('======================================================');
    console.error('Bu modül, resmi olmayan bir WhatsApp Web otomasyonu kullanır.');
    console.error('Meta (WhatsApp Business Terms), veri kazıma ve izinsiz');
    console.error('otomasyon kullanımını yasaklayabilir ve cihazınız');
    console.error('veya numaranız banlanabilir.\n');
    console.error('Kullanım risklerini kabul ediyorsanız, bu scripti');
    console.error('şu bayrakla çalıştırın: --accept-tos');
    console.error('======================================================\n');
    process.exit(1);
}

// ─────────────────────────────────────────────
//  WhatsApp Client Başlatma (Bağımsız Cihaz - LocalAuth)
// ─────────────────────────────────────────────
const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: path.join(__dirname, '.wwebjs_auth')
    }),
    authTimeoutMs: 120000,
    qrMaxRetries: 10,
    webVersionCache: {
        type: 'remote',
        remotePath: 'https://raw.githubusercontent.com/wppconnect-team/wa-version/main/html/2.3000.1014-alpha.html',
    },
    puppeteer: {
        timeout: 120000,
        headless: true, // Sunucu ortamında çalışan UI botu için zorunlu
        args: chromiumArgs,
        userAgent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36'
    }
});

// Küresel Hata Yakalayıcılar
process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Beklenmeyen Rejection:', reason);
});

process.on('uncaughtException', (err) => {
    console.error('❌ Beklenmeyen İstisna:', err);
});

client.on('qr', (qr) => {
    currentState = 'QR_READY';
    lastQr = qr;
    console.log('\n=========================================');
    console.log('📱 WhatsApp Web Bağlantısı Bekleniyor');
    console.log('=========================================');
    console.log('Lütfen telefonunuzdan WhatsApp uygulamasını açın:');
    console.log('1. Ayarlar > Bağlı Cihazlar menüsüne girin.');
    console.log('2. "Cihaz Bağla" seçeneğine dokunun.');
    console.log('3. Aşağıdaki QR Kodu taratın.\n');
    qrcodeTerminal.generate(qr, { small: true });
    
    // QR kodu dosyaya kaydet (Web UI için)
    QRCode.toFile(path.join(__dirname, 'qr.png'), qr, (err) => {
        if (err) console.error('QR Dosya Kayıt Hatası:', err);
        else console.log('✅ QR Kod qr.png olarak kaydedildi.');
    });
});

client.on('ready', () => {
    currentState = 'CONNECTED';
    lastQr = null;
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
        const flaskAlive = await isFlaskAlive();
        if (flaskProcess || flaskAlive) {
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
    const flaskAlive = await isFlaskAlive();
    if (!flaskProcess && !flaskAlive) {
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
    msg.reply('⏳ Görev alındı, çalışıyorum...');

    try {
        const response = await axios.post('http://127.0.0.1:5000/whatsapp-local', {
            text: cleanedText,
            from: msg.from
        }, {
            timeout: 300000, // 5 dakika timeout
            headers: API_KEY ? { 'X-API-Key': API_KEY } : {}
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

// ─────────────────────────────────────────────
//  Push Message API (Flask → Node.js → WhatsApp)
//  Flask ajan çalışırken ara durum bilgisi gönderir
// ─────────────────────────────────────────────
const pushApp = express();
pushApp.use(express.json());

pushApp.post('/push-message', (req, res) => {
    const { to, text } = req.body;
    if (!to || !text) {
        return res.status(400).json({ error: 'to ve text gerekli' });
    }

    client.sendMessage(to, text)
        .then(() => {
            console.log(`[Push] ✅ Mesaj gönderildi → ${to.split('@')[0]}`);
            res.json({ ok: true });
        })
        .catch(err => {
            console.error(`[Push] ❌ Hata:`, err.message);
            res.status(500).json({ error: err.message });
        });
});

pushApp.get('/status', (req, res) => {
    res.json({ status: currentState });
});

pushApp.get('/qr', (req, res) => {
    res.json({ qr: lastQr });
});

const PUSH_PORT = 3001;
const server = pushApp.listen(PUSH_PORT, () => {
    console.log(`📡 WhatsApp İletişim API dinleniyor: http://localhost:${PUSH_PORT}`);
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`❌ Hata: Port ${PUSH_PORT} zaten kullanımda!`);
    } else {
        console.error('❌ Sunucu Hatası:', err);
    }
    process.exit(1);
});

// ─────────────────────────────────────────────
//  Session Lock Temizliği (Puppeteer Çakışmalarını Önlemek İçin)
// ─────────────────────────────────────────────
const sessionDir = path.join(__dirname, '.wwebjs_auth', 'session');
try {
    const lockFiles = ['SingletonLock', 'SingletonCookie', 'SingletonSocket'];
    for (const file of lockFiles) {
        const filePath = path.join(sessionDir, file);
        fs.rmSync(filePath, { force: true });
    }
    console.log('🧹 Eski session kilitleri (varsa) temizlendi.');
} catch (e) {
    console.error('⚠️ Kilit temizleme hatası:', e.message);
}

// WhatsApp client'ı başlat
console.log('🚀 WhatsApp Client başlatılıyor...');
client.initialize().catch(err => {
    console.error('❌ Client Başlatma Hatası:', err);
});

// Sistemi güvenli kapatmak
process.on('SIGINT', () => {
    if (flaskProcess) {
        console.log('Çekirdek sunucu kapatılıyor...');
        try { process.kill(-flaskProcess.pid); } catch (e) { }
    }
    process.exit();
});
