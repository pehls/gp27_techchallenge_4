import streamlit as st
import src.generate_graphs as generate_graphs
import src.get_data as get_data

st.title('Indicadores')

st.info('Complementando nossa análise, utilizamos os três indicadores abaixo para entender o comportamento dos dados')

tab_bb, tab_rsi, tab_macd = st.tabs(['Bolling Bands', 'RSI', 'MACD'])

with tab_bb:
    st.markdown("""
        A utilização das Bandas de Bollinger em nosso modelo visa identificar momentos de alta volatilidade e possíveis pontos de reversão. 
        Isso é essencial para entender os cenários em que o mercado pode estar passando por mudanças substanciais, permitindo-nos ajustar nossas previsões para refletir a volatilidade atual.
    """)

    st.pyplot(
        generate_graphs._grafico_bb(get_data._calculate_bollinger_bands()),
        use_container_width=True,
    )

    st.markdown("""
        Tomando como exemplo o gráfico de 2022, a partir da análise com bandas de bollinger podem nos indicar sinais de reversão da tendência atual. 
        Quando o volume de fechamento ultrapassa uma das bandas, isso pode indicar uma possível reversão da tendência atual. 
        Por exemplo, uma quebra acima da Banda de Bollinger superior pode sugerir uma possível reversão para baixa volatilidade ou uma mudança na tendência de baixa para alta.
    """)

with tab_rsi:
    st.markdown("""
        Optamos por utilizá-lo para auxiliar o nosso modelo a nos fornecer insights sobre a força de uma tendência, 
        indicando em quais momentos a IBOVESPA estava em momentos de "sobrecomprado" (potencialmente devido a uma alta excessiva) 
        ou "sobrevendido" (possivelmente causado por uma queda excessiva nos preços).

        Ao incorporar o RSI em nosso modelo, buscamos capturar possíveis reversões de tendência com base nas condições de sobrecompra e sobrevenda, 
        permitindo que nosso modelo identifique esses pontos e ajuste suas previsões.
    """)

    st.pyplot(
        generate_graphs._grafico_rsi(get_data._calculate_rsi()),
        use_container_width=True,
    )

    st.markdown("""
        O RSI é varia entre 0 e 100. Tradicionalmente, valores acima de 70 indicam uma condição de sobrecompra, 
        sugerindo que o ativo pode estar prestes a sofrer uma reversão de baixa. 
        Valores abaixo de 30 indicam uma condição de sobrevenda, sugerindo que o ativo pode estar prestes a se recuperar.
    """)

with tab_macd:
    st.markdown("""
        Utilizamos o MACD para obter informações sobre a relação das médias móveis, indicando a força e a direção de uma tendência atual.  
        Além disso, conseguimos obter o dado de velocidade com que ela está se desenvolvendo. 
        Isso nos permite ajustar nossas previsões com base em mudanças abruptas ou gradualidades na evolução dos preços, 
        contribuindo para uma compreensão mais abrangente da dinâmica do mercado.
    """)

    st.pyplot(
        generate_graphs._grafico_macd(get_data._calculate_macd()),
        use_container_width=True,
    )

    st.markdown("""
        #Quando há cruzamentos entre o MACD e a linha de sinal (o MACD cruza acima ou abaixo da linha de sinal) e esse cruzamento 
        coincide com movimentos significativos no preço de fechamento, isso pode ser considerado um sinal mais forte. 
        Por exemplo, se o MACD cruzar acima da linha de sinal ao mesmo tempo em que o preço de fechamento está subindo, isso pode indicar uma possível tendência de alta mais forte.
    """)