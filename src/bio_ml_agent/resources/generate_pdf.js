const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    try {
        const browser = await puppeteer.launch({
            executablePath: '/usr/bin/google-chrome',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        const page = await browser.newPage();
        const filePath = 'file://' + path.resolve(__dirname, 'deep_report.html');
        console.log('Loading:', filePath);
        await page.goto(filePath, { waitUntil: 'networkidle0' });
        await page.pdf({
            path: 'Deep_Research_Defense_Mechanisms.pdf',
            format: 'A4',
            printBackground: true,
            margin: { top: '10mm', right: '10mm', bottom: '10mm', left: '10mm' }
        });
        await browser.close();
        console.log('PDF generated successfully: Deep_Research_Defense_Mechanisms.pdf');
    } catch (err) {
        console.error('Error generating PDF:', err);
        process.exit(1);
    }
})();
