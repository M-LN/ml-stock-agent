mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor = \"#FF4B4B\"\n\
backgroundColor = \"#0E1117\"\n\
secondaryBackgroundColor = \"#262730\"\n\
textColor = \"#FAFAFA\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml

echo "\
newsapi_key = \"$NEWSAPI_KEY\"\n\
" > ~/.streamlit/secrets.toml
