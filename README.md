# Finance Tracker

מערכת מעקב אחר מניות וביטקוין עם תחזיות ותבניות טכניות.

## תכונות

- מעקב אחר מניות וביטקוין בזמן אמת
- זיהוי תבניות טכניות (Cup and Handle, Head and Shoulders, Double Top)
- תחזיות מחיר ל-3 ימים, שבוע וחודש
- ממשק משתמש דו-לשוני (עברית/אנגלית)
- גרפים אינטראקטיביים
- חיזוי מחירי יעד לתבניות פעילות

## התקנה

1. התקן את Python 3.10 או גרסה חדשה יותר
2. שכפל את המאגר:
```bash
git clone https://github.com/[your-username]/finance_tracker.git
cd finance_tracker
```

3. צור והפעל סביבה וירטואלית:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

4. התקן את הדרישות:
```bash
pip install -r requirements.txt
```

5. הפעל את השרת:
```bash
python app.py
```

6. פתח את הדפדפן בכתובת `http://localhost:5000`

## טכנולוגיות

- Python Flask
- Plotly
- yfinance
- NumPy
- Pandas
- SciPy

## רישיון

MIT License
