import webview
import json

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Tournament Simulator</title>
    <style>
        :root {
            --bg: #ffffff;
            --fg: #000000;
            --border: #e0e0e0;
            --champ-bg: #ffd700;
            --winner-bg: #e8f5e9;
            --bye-bg: #f5f5f5;
            --tab-bg: #f8f9fa;
            --tab-active: #ffffff;
            --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            font-family: var(--font);
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--fg);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        /* ----- DIALOG ----- */
        #dialog-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.6);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .dialog {
            background: var(--bg);
            padding: 30px;
            border-radius: 12px;
            width: 420px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .dialog h2 { margin-top: 0; margin-bottom: 20px; font-size: 24px; font-weight: 800;}
        .row { margin-bottom: 15px; }
        .row label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 14px; color: #333; }
        .row input[type="text"], .row input[type="number"], .row select {
            width: 100%; padding: 10px; border: 1px solid #ccc;
            border-radius: 6px; box-sizing: border-box; font-size: 14px;
        }
        .help-text { font-size: 13px; color: #666; margin-top: 6px; display: block; font-style: italic; }
        
        .radio-group { display: flex; gap: 15px; }
        .radio-group label { font-weight: normal; display: flex; align-items: center; gap: 5px; color: #444;}

        .btn-row { display: flex; justify-content: flex-end; gap: 10px; margin-top: 25px; }
        button {
            padding: 10px 20px; font-weight: 700; font-family: var(--font);
            cursor: pointer; border-radius: 6px; border: 2px solid var(--fg);
            transition: all 0.2s; font-size: 14px;
        }
        .btn-primary { background: var(--fg); color: var(--bg); }
        .btn-primary:hover { background: #333; }
        .btn-secondary { background: var(--bg); color: var(--fg); border-color: #ccc; }
        .btn-secondary:hover { background: #f0f0f0; }

        /* ----- MAIN UI ----- */
        #main-ui {
            display: none;
            flex-direction: column;
            height: 100%;
        }

        .header {
            padding: 20px 30px;
            background: #ffffff;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header-info h1 { margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
        .header-info p { margin: 5px 0 0; font-size: 14px; color: #666; }

        /* ----- TABS ----- */
        .tabs-container {
            display: flex;
            background: var(--tab-bg);
            border-bottom: 1px solid var(--border);
            padding: 0 20px;
            overflow-x: auto;
        }
        .tab {
            padding: 15px 25px;
            font-size: 14px;
            font-weight: 600;
            color: #666;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }
        .tab:hover { color: #000; }
        .tab.active {
            color: #000;
            border-bottom-color: #000;
            background: var(--tab-active);
        }
        .tab.disabled {
            color: #ccc;
            cursor: not-allowed;
            pointer-events: none;
        }

        /* ----- MODAL ----- */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal-content {
            background: #fff;
            padding: 30px;
            border-radius: 12px;
            width: 500px;
            max-width: 90%;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .modal-content h3 {
            margin-top: 0;
            color: #333;
        }
        .disambig-row {
            margin-bottom: 15px;
        }
        .disambig-row label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
            color: #555;
        }
        .disambig-row select {
            width: 100%;
            padding: 8px;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 14px;
        }

        /* ----- BRACKET CARDS ----- */
        #screens {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .bracket-container {
            flex: 1;
            overflow: auto;
            padding: 20px 40px;
            background: #fff;
            display: none;
            scroll-behavior: smooth;
        }
        .bracket-container.active { display: block; }

        /* This inner wrapper sizes itself to the tallest column (Round 1).
           All other columns stretch to match, enabling the triangle effect. */
        .bracket-inner {
            display: inline-flex;
            align-items: stretch;
            min-height: 100%;
        }

        .tree-column {
            display: flex;
            flex-direction: column;
            min-width: 280px;
            flex-shrink: 0;
            position: relative;
        }

        .column-header {
            text-align: center;
            font-size: 13px;
            font-weight: 800;
            color: #111;
            padding: 15px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 1px solid #eee;
            margin-bottom: 0;
            position: sticky;
            top: 0;
            background: #fff;
            z-index: 10;
        }

        /* The Triangle Spacing System */
        .match-node {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            flex: 1; /* This ensures equal vertical distribution */
            position: relative;
            padding: 10px 0;
        }

        .match-card {
            background: #fff;
            border-radius: 4px;
            width: 240px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            position: relative;
            border: 1px solid #ddd;
            z-index: 2;
        }
        
        /* BNP Style Player Rows */
        .player-row {
            height: 36px;
            padding: 0 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
            border-bottom: 1px solid #f0f0f0;
        }
        .player-row:last-child { border-bottom: none; }
        .player-row .name { font-weight: 500; color: #111; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .player-row .seed { font-size: 10px; color: #888; font-weight: 700; background: #f8f8f8; padding: 2px 5px; border-radius: 2px; }
        .player-row.winner .name { font-weight: 800; }
        .player-row.winner .score { font-weight: 800; color: #000; }
        .player-row .score { font-size: 12px; color: #666; width: 60px; text-align: right; }
        .player-row.bye .name { font-style: italic; color: #ccc; }

        /* Connector Triangle Lines */
        .match-node::after {
            content: '';
            position: absolute;
            right: 0;
            top: 50%;
            width: 40px;
            height: 50%;
            border-right: 1.5px solid #ccc;
            z-index: 1;
        }
        .match-node.top-match::after {
            top: 50%;
            border-top: 1.5px solid #ccc;
        }
        .match-node.bottom-match::after {
            top: 0;
            height: 50%;
            border-bottom: 1.5px solid #ccc;
        }
        .match-node.no-connect::after { display: none; }

        /* Inlet line from previous round */
        .match-node::before {
            content: '';
            position: absolute;
            left: -40px;
            top: 50%;
            width: 40px;
            border-top: 1.5px solid #ccc;
            z-index: 1;
        }
        .match-node.first-round::before { display: none; }
        /* The line going into the next round */
        .match-node.has-next-round::before {
            content: '';
            position: absolute;
            right: -60px;
            top: 50%;
            width: 30px;
            border-top: 2px solid #ccc;
        }

        .match-header {
            background: #f8f9fa;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: bold;
            color: #888;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .player-row {
            display: flex;
            align-items: center;
            height: 40px;
            border-bottom: 1px solid #f0f0f0;
            position: relative;
            padding: 0 10px;
            z-index: 1; /* Above connector lines */
            background: #fff;
            border-radius: 8px;
        }
        .player-row:last-child { border-bottom: none; }

        .name {
            flex: 1;
            font-size: 14px;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .name input {
            width: 100%; height: 100%;
            border: none; outline: none;
            font-size: 14px; font-weight: 600;
            font-family: inherit;
            background: transparent;
        }
        .name input::placeholder { color: #ccc; font-weight: normal; }

        .seed {
            font-size: 11px;
            font-weight: 600;
            color: #aaa;
            margin-left: 8px;
            text-align: right;
            min-width: 20px;
        }
        .name input {
            width: 100%; height: 100%;
            border: none; outline: none;
            font-size: 15px; font-weight: 500;
            font-family: inherit;
        }
        .name input::placeholder { color: #ccc; font-weight: normal; }

        .score {
            padding: 0 12px;
            font-size: 13px;
            font-weight: 600;
            color: #555;
            text-align: right;
        }
        
        .winner-check {
            position: absolute;
            left: -8px;
            color: transparent;
            font-size: 14px;
        }

        .player-row.bye { background: var(--bye-bg); }
        .player-row.bye .name { color: #888; font-weight: 600; font-size: 13px; }
        .player-row.winner { background: var(--winner-bg); }
        .player-row.winner .winner-check { position: static; color: #2e7d32; margin-left: 8px; }
        .player-row.winner .name { font-weight: 700; color: #1b5e20; }
        
        .champion-card {
            background: var(--champ-bg);
            border: 2px solid #e6c200;
            transform: scale(1.05);
            margin-top: 40px;
        }
        .champion-card .match-header { background: rgba(255,255,255,0.4); border-bottom-color: rgba(0,0,0,0.1); color: #8a7400; }
        .champion-card .player-row { height: 60px; border: none; }
        .champion-card .name { font-size: 20px; font-weight: 800; color: #5c4d00; text-align: center; }

        /* ----- ATP FINALS TABLES ----- */
        .group-table {
            width: 100%;
            border-collapse: collapse;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        .group-table th {
            background: #f0f0f0;
            padding: 10px 15px;
            text-align: left;
            font-size: 13px;
            font-weight: bold;
            color: #333;
            border-bottom: 1px solid #ccc;
        }
        .group-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }
        .group-table tr:last-child td { border-bottom: none; }
        .group-table tr.qualify { background: #e8f5e9; }
        .group-table tr.qualify td:first-child { font-weight: bold; color: #1b5e20; }
        
        .atp-group-container {
            width: 100%;
            max-width: 800px;
            margin-bottom: 40px;
        }
        .atp-group-title {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 10px;
            color: #333;
        }

    </style>
</head>
<body>

    <!-- SETUP DIALOG -->
    <div id="dialog-overlay">
        <div class="dialog">
            <h2>Tournament Setup</h2>
            
            <div class="row">
                <label>Tournament Level</label>
                <select id="p-level" onchange="updateByes()">
                    <option value="G">Grand Slam</option>
                    <option value="M" selected>Masters 1000</option>
                    <option value="A">ATP 500/250</option>
                    <option value="F">ATP Finals (8-player RR)</option>
                </select>
            </div>

            <div class="row">
                <label>Number of Players</label>
                <input type="number" id="p-count" value="96" min="2" oninput="updateByes()">
                <span class="help-text" id="bye-info"></span>
            </div>

            <div class="row">
                <label>Surface</label>
                <div class="radio-group">
                    <label><input type="radio" name="surf" value="Hard" checked> Hard</label>
                    <label><input type="radio" name="surf" value="Clay"> Clay</label>
                    <label><input type="radio" name="surf" value="Grass"> Grass</label>
                </div>
            </div>

            <div class="row">
                <label>Tournament Name</label>
                <input type="text" id="p-name" placeholder="e.g. Indian Wells">
            </div>

            <div class="btn-row">
                <button class="btn-secondary" onclick="cancel()">Cancel</button>
                <button class="btn-primary" onclick="createBracket()">Create Bracket →</button>
            </div>
        </div>
    </div>

    <!-- MAIN UI -->
    <div id="main-ui">
        <div class="header">
            <div class="header-info">
                <h1 id="h-name">Tournament</h1>
                <p id="h-details">Loading...</p>
            </div>
            <div>
                <span id="status-text" style="color: #666; margin-right: 15px; font-size: 14px; font-weight: 500;"></span>
                <button class="btn-primary" id="btn-sim" onclick="simulate()">▶ Simulate Tournament</button>
            </div>
        </div>
        
        <div class="tabs-container" id="tabs">
            <!-- Rendered via JS -->
        </div>

        <div id="screens">
            <!-- Bracket containers rendered via JS -->
        </div>
    </div>

    <!-- DISAMBIGUATION MODAL -->
    <div class="modal-overlay" id="disambig-modal">
        <div class="modal-content">
            <h3>Disambiguate Players</h3>
            <p style="font-size: 14px; color: #666; margin-bottom: 20px;">
                Multiple players matched the initials you provided. Please select the correct player for each:
            </p>
            <div id="disambig-list"></div>
            <div class="btn-row" style="margin-top: 25px;">
                <button class="btn-secondary" onclick="document.getElementById('disambig-modal').style.display='none'">Cancel</button>
                <button class="btn-primary" onclick="confirmDisambiguation()">Confirm & Simulate →</button>
            </div>
        </div>
    </div>

    <script>
        let setupData = null;
        let seedDist = [];
        let simulationResults = null;

        function nextPowerOf2(n) {
            return n <= 1 ? 2 : Math.pow(2, Math.ceil(Math.log2(n)));
        }

        function getRoundName(n) {
            if (n === 2) return 'Final';
            if (n === 4) return 'Semi-Finals';
            if (n === 8) return 'Quarter-Finals';
            return `Round of ${n}`;
        }

        function updateByes() {
            const level = document.getElementById('p-level').value;
            const pCountInput = document.getElementById('p-count');
            const info = document.getElementById('bye-info');
            
            if (level === 'F') {
                pCountInput.value = 8;
                pCountInput.disabled = true;
                info.innerText = `ATP Finals format restricts draw to exactly 8 players.`;
                return;
            } else {
                pCountInput.disabled = false;
            }

            const n = parseInt(pCountInput.value) || 0;
            const draw = nextPowerOf2(n);
            
            if (n >= 2 && draw > n) {
                const byes = draw - n;
                info.innerText = `ATP Rule: Draw size expands to ${draw}. Top ${byes} seeds receive an automatic first-round BYE.`;
            } else {
                info.innerText = `Perfect draw of ${n}. No byes needed.`;
            }
        }
        
        updateByes();

        function cancel() {
            pywebview.api.cancel();
        }

        async function createBracket() {
            const level = document.getElementById('p-level').value;
            const n = parseInt(document.getElementById('p-count').value);
            
            if (!n || n < 2) return alert("Need at least 2 players");
            
            const draw = nextPowerOf2(n);
            const byes = draw - n;
            const surf = document.querySelector('input[name="surf"]:checked').value;
            let name = document.getElementById('p-name').value;
            if (!name) name = `${surf} Tournament`;

            setupData = { 
                num_players: level === 'F' ? 8 : n, 
                draw_size: level === 'F' ? 8 : draw, 
                num_byes: level === 'F' ? 0 : byes, 
                surface: surf, 
                level: level, 
                name: name 
            };
            
            document.getElementById('dialog-overlay').style.display = 'none';
            document.getElementById('main-ui').style.display = 'flex';
            document.getElementById('h-name').innerText = name;
            
            if (level === 'F') {
                document.getElementById('h-details').innerText = `${surf} · ATP Finals · 8 Players (Round Robin)`;
                renderAtpFinalsInput();
            } else {
                document.getElementById('h-details').innerText = `${surf} · ${level} · Draw: ${draw} · Players: ${n} · Byes: ${byes}`;
                seedDist = await pywebview.api.get_seed_distribution(draw);
                renderKnockoutInput();
            }
        }

        function switchTab(rIdx) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.bracket-container').forEach(c => c.classList.remove('active'));
            
            const tr = document.getElementById(`tab-${rIdx}`);
            const sr = document.getElementById(`screen-${rIdx}`);
            if(tr) tr.classList.add('active');
            if(sr) sr.classList.add('active');
        }

        // =====================================================================
        // KNOCKOUT BRACKET SYSTEM
        // =====================================================================

        function renderKnockoutInput() {
            const roundsCount = Math.log2(setupData.draw_size);
            const tabsEl = document.getElementById('tabs');
            const screensEl = document.getElementById('screens');
            
            // Single Tab for standard Draw
            tabsEl.innerHTML = `<div class="tab active" id="tab-0">Tournament Draw</div>`;
            screensEl.innerHTML = `<div class="bracket-container active" id="screen-0"><div class="bracket-inner" id="bracket-inner"></div></div>`;
            
            const container = document.getElementById('bracket-inner');
            
            // Generate all columns in one container
            for (let r = 0; r <= roundsCount; r++) {
                const playersInThisRound = setupData.draw_size / Math.pow(2, r);
                const matchesInThisRound = playersInThisRound / 2;
                const isChampion = (r === roundsCount);
                const rName = isChampion ? "Champion" : getRoundName(playersInThisRound);

                const col = document.createElement('div');
                col.className = 'tree-column';
                col.id = `round-col-${r}`;
                
                col.innerHTML = `<div class="column-header">${rName}</div>`;
                
                if (r === 0) {
                    // Round 1 Input
                    for (let i = 0; i < seedDist.length; i += 2) {
                        const s1 = seedDist[i];
                        const s2 = seedDist[i+1];
                        const matchNum = (i/2) + 1;
                        
                        const nodeClass = (matchNum % 2 !== 0) ? 'top-match' : 'bottom-match';
                        const node = document.createElement('div');
                        node.className = `match-node first-round ${nodeClass}`;
                        
                        const card = document.createElement('div');
                        card.className = 'match-card';
                        card.innerHTML = `
                            ${buildPlayerRowHTML(s1, setupData.num_players)}
                            ${buildPlayerRowHTML(s2, setupData.num_players)}
                        `;
                        node.appendChild(card);
                        col.appendChild(node);
                    }
                } else if (!isChampion) {
                    // Empty shells for future rounds
                    for (let i = 0; i < matchesInThisRound; i++) {
                        const matchNum = i + 1;
                        const nodeClass = (matchNum % 2 !== 0) ? 'top-match' : 'bottom-match';
                        const isLastMatch = (matchesInThisRound === 1);
                        
                        const node = document.createElement('div');
                        node.className = `match-node ${isLastMatch ? 'no-connect' : nodeClass}`;
                        
                        const card = document.createElement('div');
                        card.className = 'match-card';
                        card.id = `node-r${r}-m${i}`;
                        card.innerHTML = `
                            <div class="player-row"><div class="name" style="color:#ddd">TBD</div></div>
                            <div class="player-row"><div class="name" style="color:#ddd">TBD</div></div>
                        `;
                        node.appendChild(card);
                        col.appendChild(node);
                    }
                } else {
                    // Champion Shell
                    const node = document.createElement('div');
                    node.className = `match-node no-connect`;
                    const card = document.createElement('div');
                    card.className = 'match-card champion-card';
                    card.id = `node-champ`;
                    card.style.border = '2px solid #b39b00';
                    card.innerHTML = `
                        <div class="player-row" style="background:#fffbe6; justify-content:center; height:60px;">
                            <div class="name" style="color:#b39b00; font-weight:800; text-align:center; font-size:18px;">CHAMPION</div>
                        </div>
                    `;
                    node.appendChild(card);
                    col.appendChild(node);
                }
                
                container.appendChild(col);
            }
        }

        // =====================================================================
        // ATP FINALS (ROUND ROBIN) SYSTEM
        // =====================================================================
        
        function renderAtpFinalsInput() {
            const tabsEl = document.getElementById('tabs');
            const screensEl = document.getElementById('screens');
            tabsEl.innerHTML = ''; screensEl.innerHTML = '';
            
            const tabsData = ['Group Input', 'Group Stage Matches', 'Standings', 'Semi-Finals', 'Final'];
            
            for (let i = 0; i < tabsData.length; i++) {
                const tab = document.createElement('div');
                tab.className = 'tab' + (i === 0 ? ' active' : ' disabled');
                tab.id = `tab-${i}`;
                tab.innerText = tabsData[i];
                tab.onclick = () => { if (!tab.classList.contains('disabled')) switchTab(i); };
                tabsEl.appendChild(tab);

                const screen = document.createElement('div');
                screen.className = 'bracket-container' + (i === 0 ? ' active' : '');
                screen.id = `screen-${i}`;
                screensEl.appendChild(screen);
            }
            
            const inputScreen = document.getElementById('screen-0');
            
            const createGroupInput = (title, seeds) => {
                let html = `<div class="atp-group-container"><div class="atp-group-title">${title}</div><div class="round-grid">`;
                for (let seed of seeds) {
                    html += `
                        <div class="match-card">
                            <div class="player-row">
                                <div class="seed">${seed}</div>
                                <div class="name">
                                    <input type="text" id="in_name_${seed}" placeholder="Player Name...">
                                </div>
                            </div>
                        </div>
                    `;
                }
                html += `</div></div>`;
                return html;
            };
            
            // Group A: 1, 4, 5, 8
            inputScreen.innerHTML += createGroupInput("Group A", [1, 4, 5, 8]);
            // Group B: 2, 3, 6, 7
            inputScreen.innerHTML += createGroupInput("Group B", [2, 3, 6, 7]);
        }

        // =====================================================================
        // SHARED SIMULATION DISPATCHER
        // =====================================================================

        function buildPlayerRowHTML(seed, totalPlayers) {
            if (seed > totalPlayers) {
                return `
                    <div class="player-row bye">
                        <div class="name">BYE</div>
                    </div>
                `;
            }
            return `
                <div class="player-row">
                    <div class="name">
                        <input type="text" id="in_name_${seed}" placeholder="Player Name...">
                    </div>
                    ${seed ? `<div class="seed">${seed}</div>` : ''}
                </div>
            `;
        }

        // Store unresolved ambiguities temporarily
        let pendingAmbiguities = {};

        async function simulate() {
            document.getElementById('status-text').innerText = 'Validating names...';
            const btn = document.getElementById('btn-sim');
            btn.disabled = true;
            
            // Collect all input names
            let namesToValidate = [];
            document.querySelectorAll('input[id^="in_name_"]').forEach(inp => {
                const val = inp.value.trim();
                if (val && val.toUpperCase() !== 'BYE') {
                    namesToValidate.push(val);
                }
            });

            if (namesToValidate.length < 2) {
                document.getElementById('status-text').innerText = 'Need at least 2 players!';
                btn.disabled = false;
                return;
            }

            // Call Python API to validate names
            try {
                pendingAmbiguities = await pywebview.api.validate_names(namesToValidate);
                
                if (Object.keys(pendingAmbiguities).length > 0) {
                    // Show disambiguation modal
                    document.getElementById('status-text').innerText = 'Waiting for clarification...';
                    btn.disabled = false;
                    showDisambiguationModal();
                    return;
                }
            } catch (e) {
                console.error("Validation error:", e);
            }

            // If everything is clean, proceed
            executeSimulation();
        }

        function showDisambiguationModal() {
            const listEl = document.getElementById('disambig-list');
            listEl.innerHTML = '';
            
            for (const [inputName, matches] of Object.entries(pendingAmbiguities)) {
                const row = document.createElement('div');
                row.className = 'disambig-row';
                
                // Use a safer ID for DOM mapping
                const safeId = "disambig_" + btoa(inputName).replace(/=/g, '');
                
                let selectHtml = `<select id="${safeId}">`;
                matches.forEach(m => {
                    selectHtml += `<option value="${m}">${m}</option>`;
                });
                selectHtml += `</select>`;
                
                row.innerHTML = `<label>For "${inputName}":</label>${selectHtml}`;
                listEl.appendChild(row);
            }
            
            document.getElementById('disambig-modal').style.display = 'flex';
        }

        function confirmDisambiguation() {
            // Overwrite the input fields with the selected full names
            const inputs = document.querySelectorAll('input[id^="in_name_"]');
            
            inputs.forEach(inp => {
                const val = inp.value.trim();
                if (pendingAmbiguities[val]) {
                    const safeId = "disambig_" + btoa(val).replace(/=/g, '');
                    const mappedSelect = document.getElementById(safeId);
                    if (mappedSelect) {
                        inp.value = mappedSelect.value;
                    }
                }
            });
            
            document.getElementById('disambig-modal').style.display = 'none';
            pendingAmbiguities = {};
            executeSimulation();
        }

        async function executeSimulation() {
            document.getElementById('status-text').innerText = 'Simulating tournament...';
            const btn = document.getElementById('btn-sim');
            btn.disabled = true;

            const players = [];
            let isValid = true;
            
            if (setupData.level === 'F') {
                for (let sd = 1; sd <= 8; sd++) {
                    const el = document.getElementById(`in_name_${sd}`);
                    if (!el) continue;
                    const name = el.value.trim();
                    if (name) {
                        players.push({ name: name, seed: sd, year: null });
                    }
                }
            } else {
                for (let sd of seedDist) {
                    if (sd <= setupData.num_players) {
                        const el = document.getElementById(`in_name_${sd}`);
                        if (!el) continue;
                        const name = el.value.trim();
                        if (name) {
                            players.push({ name: name, seed: sd, year: null });
                        }
                    }
                }
            }

            if (players.length < 2) {
                document.getElementById('status-text').innerText = 'Missing player names!';
                btn.disabled = false;
                return;
            }
            
            if (setupData.level === 'F' && players.length !== 8) {
                document.getElementById('status-text').innerText = 'ATP Finals requires 8 players!';
                btn.disabled = false;
                return;
            }
            
            document.querySelectorAll('input[id^="in_name_"]').forEach(el => el.disabled = true);

            try {
                const res = await pywebview.api.run_simulation(setupData, players);
                if (res.error) throw new Error(res.error);
                
                simulationResults = res;
                
                if (res.type === 'atp_finals') {
                    renderAtpResultsScreens();
                } else {
                    renderKnockoutResultsScreens();
                }
                
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('disabled'));
                document.getElementById('status-text').innerText = "Simulation Complete!";
                btn.disabled = false;
                btn.innerText = "▶ Restart Simulator";
                btn.onclick = () => window.location.reload();
                
                // For trees, we don't switch tabs, we just show the results in the tree columns
                // but let's scroll to the right? or just stay put.
            } catch (err) {
                alert("Error: " + err);
                document.getElementById('status-text').innerText = "Simulation Failed";
                btn.disabled = false;
                document.querySelectorAll('input').forEach(el => el.disabled = false);
            }
        }

        function renderKnockoutResultsScreens() {
            const history = simulationResults.bracket_history;
            const roundsCount = history.length;

            for (let r = 0; r < roundsCount; r++) {
                const matchesRow = history[r].results;
                
                for (let m = 0; m < matchesRow.length; m++) {
                    const match = matchesRow[m];
                    const p1 = match.player1;
                    const p2 = match.player2;
                    const winner = match.winner;
                    
                    if (p1.name === 'BYE' && p2.name === 'BYE') continue;

                    let scoreText = match.score || '';
                    if (scoreText === 'BYE') scoreText = '';

                    const isP1Winner = (winner.name === p1.name);
                    const isP2Winner = (winner.name === p2.name);

                    // Find existing card shell
                    const card = r === 0 
                        ? document.querySelector(`#round-col-0 .match-node:nth-child(${m + 2}) .match-card`)
                        : document.getElementById(`node-r${r}-m${m}`);

                    if (card) {
                        card.innerHTML = `
                            ${buildResultRowHTML(p1, isP1Winner, isP1Winner ? scoreText : '')}
                            ${buildResultRowHTML(p2, isP2Winner, isP2Winner ? scoreText : '')}
                        `;
                    }
                }
            }

            // Update Champion Card
            const finalMatch = history[history.length - 1].results[0];
            const champPlayer = finalMatch.winner;
            const champCard = document.getElementById('node-champ');
            if (champCard) {
                champCard.innerHTML = `
                    <div class="player-row" style="background:#fffbe6; justify-content:center; height:60px; border:none;">
                        <span class="seed" style="color:#b39b00; margin-right:8px;">${champPlayer.seed || ''}</span>
                        <span class="name" style="color:#b39b00; font-weight:800; font-size:20px;">${champPlayer.name}</span>
                    </div>
                `;
            }
            
            // Scroll to the end of the draw
            const scrollContainer = document.getElementById('screen-0');
            scrollContainer.scrollLeft = scrollContainer.scrollWidth;
        }

        function renderAtpResultsScreens() {
            const scrMatches = document.getElementById('screen-1');
            const scrStandings = document.getElementById('screen-2');
            const scrSF = document.getElementById('screen-3');
            const scrFinal = document.getElementById('screen-4');
            
            scrMatches.innerHTML = ''; scrStandings.innerHTML = ''; scrSF.innerHTML = ''; scrFinal.innerHTML = '';
            
            // 1) Render Round Robin Matches
            let mGrid = `<div class="round-grid">`;
            const rr = simulationResults.rr_matches;
            for (let i = 0; i < rr.length; i++) {
                const match = rr[i];
                const p1 = match.player1; const p2 = match.player2; const w = match.winner;
                const isP1 = (w.name === p1.name); 
                const isP2 = (w.name === p2.name);
                
                mGrid += `
                    <div class="match-card">
                        <div class="match-header">${i < 6 ? 'Group A Match' : 'Group B Match'}</div>
                        ${buildResultRowHTML(p1, isP1, isP1 ? match.score : '')}
                        ${buildResultRowHTML(p2, isP2, isP2 ? match.score : '')}
                    </div>
                `;
            }
            mGrid += '</div>';
            scrMatches.innerHTML = mGrid;
            
            // 2) Render Standings
            const buildTable = (title, group) => {
                let rows = '';
                for (let i=0; i<group.length; i++) {
                    const p = group[i];
                    rows += `<tr class="${i < 2 ? 'qualify' : ''}">
                        <td>${i+1}</td>
                        <td>(${p.player.seed}) ${p.player.name}</td>
                        <td>${p.wins}</td>
                        <td>${p.losses}</td>
                    </tr>`;
                }
                return `
                    <div class="atp-group-container">
                        <div class="atp-group-title">${title}</div>
                        <table class="group-table">
                            <thead><tr><th>Rank</th><th>Player</th><th>Wins</th><th>Losses</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                `;
            };
            scrStandings.innerHTML = buildTable("Group A", simulationResults.group_a) + buildTable("Group B", simulationResults.group_b);
            
            // 3) Render Semi Finals
            const history = simulationResults.bracket_history;
            const sfs = history[0].results; // [sf1, sf2]
            let sfGrid = `<div class="round-grid">`;
            for(let i=0; i<sfs.length; i++) {
                const match = sfs[i];
                sfGrid += `
                    <div class="match-card">
                        <div class="match-header">Semi-Final ${i+1}</div>
                        ${buildResultRowHTML(match.player1, match.winner.name === match.player1.name, match.winner.name === match.player1.name ? match.score : '')}
                        ${buildResultRowHTML(match.player2, match.winner.name === match.player2.name, match.winner.name === match.player2.name ? match.score : '')}
                    </div>
                `;
            }
            sfGrid += '</div>';
            scrSF.innerHTML = sfGrid;
            
            // 4) Render Final
            const finalMatch = history[1].results[0];
            const champPlayer = finalMatch.winner;
            scrFinal.innerHTML = `
                <div class="match-card">
                    <div class="match-header">Championship Match</div>
                    ${buildResultRowHTML(finalMatch.player1, finalMatch.winner.name === finalMatch.player1.name, finalMatch.winner.name === finalMatch.player1.name ? finalMatch.score : '')}
                    ${buildResultRowHTML(finalMatch.player2, finalMatch.winner.name === finalMatch.player2.name, finalMatch.winner.name === finalMatch.player2.name ? finalMatch.score : '')}
                </div>
                <div class="match-card champion-card" style="margin-top: 40px">
                    <div class="match-header" style="text-align:center">Tournament Champion</div>
                    <div class="player-row" style="justify-content:center">
                        <span class="seed" style="border:none;background:transparent;color:#8a7400">${champPlayer.seed || ''}</span>
                        <span class="name">${champPlayer.name}</span>
                    </div>
                </div>
            `;
        }

        function buildResultRowHTML(player, isWinner, scoreStr) {
            if (player.name === 'BYE') {
                return `<div class="player-row bye"><div class="name">BYE</div></div>`;
            }
            const winClass = isWinner ? 'winner' : '';
            return `
                <div class="player-row ${winClass}">
                    <div class="name">${player.name}</div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        ${scoreStr ? `<div class="score">${scoreStr}</div>` : ''}
                        ${player.seed ? `<div class="seed">${player.seed}</div>` : ''}
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

class Api:
    def __init__(self):
        self._window = None
        
    def set_window(self, window):
        self._window = window
        
    def cancel(self):
        self._window.destroy()
        
    def get_seed_distribution(self, draw_size):
        """Called from JS to get the strict ATP tournament seed distribution layout."""
        try:
            from tournament_simulator import TournamentSimulator
            sim = TournamentSimulator()
            return sim._get_seed_distribution(draw_size)
        except Exception as e:
            print("Error getting seed distribution:", e)
            return []
            
    def validate_names(self, names):
        """
        Called from JS to validate a list of names before simulation.
        Returns a dictionary of any names that need disambiguation:
        e.g. { "A Kuznetsov": ["Alex Kuznetsov", "Andrey Kuznetsov"] }
        """
        from tournament_simulator import TournamentSimulator
        sim = TournamentSimulator()
        
        ambiguities = {}
        for name in names:
            if not name or name == 'BYE': continue
            
            valid, result = sim.validate_player_name(name)
            # If valid is False, but result has a list, it's ambiguous
            if not valid and isinstance(result, list) and len(result) > 1:
                ambiguities[name] = result
        
        return ambiguities

    def run_simulation(self, setup, players):
        from tournament_simulator import TournamentSimulator
        try:
            print(f"  Simulating {setup['name']} with {len(players)} entered players...")
            sim = TournamentSimulator()
            results = sim.simulate_tournament(
                players, 
                surface=setup["surface"],
                tourney_level=setup["level"],
                use_model="average", 
                draw_size=setup["draw_size"],
                show_details=False, 
                silent=True
            )
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [clean_dict(v) for v in d]
                elif "DataFrame" in str(type(d)) or "Series" in str(type(d)):
                    return None
                # Handle numpy types that JSON rejects
                elif "numpy" in str(type(d)):
                    import numpy as np
                    if isinstance(d, np.floating):
                        return float(d)
                    elif isinstance(d, np.integer):
                        return int(d)
                    elif isinstance(d, np.ndarray):
                        return clean_dict(d.tolist())
                    return str(d)
                return d
            
            return clean_dict(results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

def launch_tournament_gui():
    """Launch the Webview GUI. Returns True if completed, False if cancelled."""
    api = Api()
    
    window = webview.create_window(
        'Tennis Tournament Simulator', 
        html=HTML_CONTENT,
        js_api=api,
        width=1100, 
        height=850,
        text_select=True
    )
    api.set_window(window)
    
    try:
        webview.start()
        return True
    except Exception as e:
        print(f"\n  [Error] Failed to start GUI engine: {e}")
        return False

if __name__ == "__main__":
    launch_tournament_gui()
