const { Keypair } = require('@solana/web3.js');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');
const fs = require('fs');

const SUFFIX = 'pump';
const TARGET_COUNT = 50;
const OUTPUT_FILE = 'vanity_keypairs.json';

if (isMainThread) {
    // Main thread - spawn workers
    const numCPUs = os.cpus().length;
    console.log(`Starting ${numCPUs} worker threads to grind "${SUFFIX}" vanity keys...`);
    console.log(`Target: ${TARGET_COUNT} keypairs`);

    let foundKeypairs = [];
    let totalAttempts = 0;
    let activeWorkers = numCPUs;
    const startTime = Date.now();

    // Load existing keypairs if any
    if (fs.existsSync(OUTPUT_FILE)) {
        try {
            foundKeypairs = JSON.parse(fs.readFileSync(OUTPUT_FILE, 'utf8'));
            console.log(`Loaded ${foundKeypairs.length} existing keypairs`);
        } catch (e) {
            console.log('Starting fresh...');
        }
    }

    for (let i = 0; i < numCPUs; i++) {
        const worker = new Worker(__filename, {
            workerData: { suffix: SUFFIX, workerId: i }
        });

        worker.on('message', (msg) => {
            if (msg.type === 'found') {
                foundKeypairs.push(msg.keypair);
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                console.log(`[${elapsed}s] Found #${foundKeypairs.length}: ${msg.keypair.pubkey}`);

                // Save progress
                fs.writeFileSync(OUTPUT_FILE, JSON.stringify(foundKeypairs, null, 2));

                if (foundKeypairs.length >= TARGET_COUNT) {
                    console.log(`\nDone! Generated ${foundKeypairs.length} vanity keypairs.`);
                    console.log(`Saved to ${OUTPUT_FILE}`);
                    process.exit(0);
                }
            } else if (msg.type === 'progress') {
                totalAttempts += msg.attempts;
            }
        });

        worker.on('error', (err) => {
            console.error(`Worker ${i} error:`, err);
        });

        worker.on('exit', () => {
            activeWorkers--;
            if (activeWorkers === 0) {
                console.log('All workers exited');
            }
        });
    }

    // Progress reporter
    setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const rate = (totalAttempts / (Date.now() - startTime) * 1000).toFixed(0);
        console.log(`[${elapsed}s] Attempts: ${totalAttempts.toLocaleString()}, Rate: ${rate}/s, Found: ${foundKeypairs.length}/${TARGET_COUNT}`);
    }, 5000);

} else {
    // Worker thread - grind keypairs
    const { suffix, workerId } = workerData;
    let attempts = 0;

    while (true) {
        const keypair = Keypair.generate();
        const pubkey = keypair.publicKey.toBase58();
        attempts++;

        if (pubkey.endsWith(suffix)) {
            parentPort.postMessage({
                type: 'found',
                keypair: {
                    pubkey: pubkey,
                    private_key: Buffer.from(keypair.secretKey).toString('base64'),
                    used: false
                }
            });
        }

        if (attempts % 100000 === 0) {
            parentPort.postMessage({ type: 'progress', attempts: 100000 });
            attempts = 0;
        }
    }
}
