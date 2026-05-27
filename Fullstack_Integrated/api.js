// File ini adalah contoh cara memanggil API AI dari Frontend (Fullstack)
// Lokasi API: https://felicia2305-robo-advisor-api.hf.space/docs

const response = await fetch("https://felicia2305-robo-advisor-api.hf.space/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    income: 50000000,
    total_expense: 10000000,
    lifestyle_expense: 2000000,
    loan_int_rate: 5.5,
    credit_score: 700,
    risk_status: 1,
    family_size: 2,
    education: 2,
    age: 30,
    budget_encoded: 1
  }),
});

const data = await response.json();
console.log(data); 
// Hasilnya: { predicted_saving_ratio: 0.8, recommendation: "Saham", ai_advice: "..." }