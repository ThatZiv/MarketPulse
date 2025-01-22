
wt -d .\server -p "Windows PowerShell" -w 0.5 -H -T "Server" -c ".venv/Scripts/Activate.ps1; python main.py"
wt -d .\client -p "Windows PowerShell" -w 0.5 -H -T "Client" -c "npm run dev -- --host"
