Hidden Machine browser task

Files in this bundle:
- index.html: the experiment interface and all JavaScript/CSS.
- machine_world_trials.json: the required adjacent trial corpus for normal mode.
- prior_participants.csv: optional ineligible-participant screening list; currently empty except for the header.
- participant_information.txt: optional placeholder participant information text.
- debrief.txt: optional debrief text.

How to run locally:
1. Keep index.html and machine_world_trials.json in the same folder.
2. Do not rely on double-clicking index.html for normal mode. Many browsers block fetch() from file:// URLs, so the JSON cannot be loaded.
3. From this folder, run a small static server:

   python3 -m http.server 8000

4. Open:

   http://localhost:8000/index.html?PROLIFIC_PID=test_pid&condition=mixed

Developer demo mode:
- You can test the embedded short demo at:

   http://localhost:8000/index.html?demo=1&debug=1&PROLIFIC_PID=test_pid

Production configuration still needed:
- Replace CONFIG.dataPipeID in index.html with the real DataPipe experiment ID.
- Replace CONFIG.prolificCompletionURL with the real Prolific completion URL.
- Replace CONFIG.prolificScreenoutURL with the real Prolific screenout URL.
- Replace participant_information.txt with approved study text if desired.
- Add prior participant IDs to prior_participants.csv if screening repeat participants.

Generated corpus note:
- machine_world_trials.json contains the required 48 observations per condition and 24 propositions.
- The observation rows were generated from the specified true stochastic hidden machine with selected_seed = 2026.
