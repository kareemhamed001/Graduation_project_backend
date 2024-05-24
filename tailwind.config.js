/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./templates/*.{html,js}"],
    theme: {
      extend: {
        colors: {
          darkPurple: "#0f0816",
          cyan:"#3DDEED",
          purple:"#BD00FF",
          darkGreen:"rgba(24, 38, 39, 0.7)",
          lightGreen:"rgba(24, 38, 39, 0.21)",
          white:"#FFFFFF",
          salate:"#94a3b8",
          darkGray:"#333333",
          LPurple:"rgba(189, 0, 255, 0.08)",
          lightPurple:"rgba(255,0,255,0.4)",
          lighterPurple:"rgba(189,0,255,0.1)",
          petroleum:"rgba(61, 222, 237, 0.08)",
          Orange:"rgba(242, 78, 30, 1)",
          BPurple:"rgba(189, 0, 255, 1)",
        },
        fontFamily: {
          Satoshi: ['Satoshi', "sans-serif"],
  
        },
        fontSize: {
          'satoshi-h1': '80px',
          'satoshi-h2': '56px',
          'satoshi-h3': '48px',
          'satoshi-h4': '30px',
          'satoshi-h5': '20px',
  
          'satoshi-p1': '18px',
          'satoshi-p2': '16px',
          'satoshi-p3': '14px',
          'satoshi-p4': '12px',
        },
      },
    },
    plugins: [],
  };