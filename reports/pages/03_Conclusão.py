import streamlit as st

st.title('Conclusão')

st.markdown("""
    Com base nos dados obtidos nas análises, podemos tirar algumas conclusões e observações sobre os modelos de previsão utilizados:

    1. Comparação dos Modelos de Previsão:
        - O **MAPE** (Erro Médio Absoluto Percentual), comparando todos os modelos testados:
            - SeasonalNaive: MAPE de aproximadamente 3.82%
            - SeasonalWindowAverage: MAPE de cerca de 3.19%
            - AutoARIMA: MAPE de cerca de 4.32%
            - Prophet (com hiperparametrização): MAPE de cerca de 6.95%

        Apesar dos dados obtidos no target do `Prophet` serem menores, mesmo após a parametrização, foram analisadas algumas características a respeito de cada um dos demais modelos para tomarmos a decisão de usar o Prophet. 

        `SeasonalNaive`: Apenas replica os valores do período sazonal anterior. Ele não leva em consideração tendências, padrões de longo prazo ou outros fatores que possam afetar as previsões.
            
        `SeasonalWindowAverage`: Não se adapta automaticamente a mudanças nos padrões dos dados. Isso pode torná-lo inadequado para séries temporais que apresentam flutuações imprevisíveis, e isso faria com que necessariamente tivéssemos que fazer ajustes contínuos no modelo para continuar tendo uma taxa de erro relativamente baixa.

        `AutoARIMA`: Tende a ser o mais complexo de configurar e usar, pois temos que selecionar hiperparâmetros críticos, como o número de diferenças (d), as ordens dos termos autorregressivos (p) e de médias móveis (q). Essa seleção pode ser complicada e envolve testar várias combinações para encontrar a melhor configuração e dado o tempo de implementação, optamos por selecionar um modelo com uma facilidade maior de implementação e hiperparametrização.

    Além disso, durante a construção do modelo com o Prophet, obtivemos:
            
    2. Melhoria na Precisão do Prophet:
        - A análise do Prophet mostra uma melhoria de 4.07% no MAPE em relação ao valor anterior (11.02% para 6.95%). Isso indica que as alterações feitas na hiperparametrização tiveram um impacto positivo na precisão do modelo.
        - Com o prophet nós temos a possibilidade de definir variáveis na modelagem para um melhor ajuste do modelo às características específicas dos dados. Isso é uma vantagem significativa, pois permite que levemos em consideração fatores externos que podem afetar as previsões e melhorem a precisão do modelo.

    E com este modelo, ao longo dos próximos meses, podemos:
            
    3. Avaliar e monitorar continuamente este modelo:
        - Os resultados atuais podem ser bons, mas é fundamental realizar avaliações contínuas e monitoramento do desempenho do modelo à medida que novos dados se tornem disponíveis, para garantir que ele continue sendo eficaz ao longo do tempo.
""")