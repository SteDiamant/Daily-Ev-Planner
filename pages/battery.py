import streamlit as st

def main():
    value = 25

    # Calculate the percentage of battery charge
    charge_percentage = value / 100.0

    # Set the height of the #charge div based on the charge percentage
    charge_height = charge_percentage * 100.0

    # Replace the placeholder {value} in the HTML code with the actual value and charge height
    html_code = f"""
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Detect Battery Status</title>
        <!-- Google Fonts -->
        <link
          href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
        <!-- Stylesheet -->
        <style>
          /* CSS code for battery status interface */
          .container {{
            width: 300px;
            margin: 50px auto;
            text-align: center;
          }}

          #battery {{
            position: relative;
            margin: 20px auto;
            width: 80px;
            height: 120px;
            border-radius: 10px;
            border: 5px solid #333;
          }}

          #charge {{
            position: absolute;
            bottom: 0;
            width: 100%;
            height: {charge_height}%; /* Set the height based on the charge percentage */
            background-color: #ff9800;
            border-radius: 0 0 10px 10px;
          }}

          #charge-level {{
            margin-top: -30px;
            font-family: 'Roboto Mono', monospace;
            font-size: 24px;
            font-weight: 500;
            color: #333;
          }}

          #charging-time {{
            font-family: 'Roboto Mono', monospace;
            font-size: 18px;
            color: #333;
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div id="battery">
            <div id="charge"></div>
            <div id="charge-level">{value}%</div> <!-- Replace {value} with the actual value -->
          </div>
          <div id="charging-time"></div>
        </div>
        <script>

        </script>
      </body>
    </html>
    """

    st.markdown(html_code, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
