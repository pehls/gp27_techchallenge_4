import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)

st.title('Modelo')


tab_modelagem_inicial, tab_resultados_iniciais, tab_conceitos, tab_variaveis_externas, tab_deploy_producao = st.tabs(['Modelagem inicial', "Resultados Iniciais", 'Conceitos', 'Variáveis Externas', "Plano - Deploy em Produção"])

df = get_data._get_modelling_data(indicators=False)

with tab_modelagem_inicial:
    st.markdown("""
        Iniciando a etapa de modelagem, optamos por experimentar o modelo Prophet, criado pela Meta/Facebook em 2017, 
        sendo um algoritmo de previsão de séries temporais que lida de forma eficiente com séries onde temos uma presença 
        forte e destacada de Sazonalidades, Feriados previamente conhecidoss e uma tendência de crescimento destacada.
        O mesmo define a série temporal tendo três componentes principais, tendência (g), sazonalidade (s) e feriados (h), 
        combinados na seguinte equação:

        `y(t)=g(t)+s(t)+h(t)+εt`, onde εt é o termo de erro, basicamente.
        
        Iniciaremos com um modelo simples, para verificar seu desempenho, e partiremos para conceitos mais elaborados, 
        chegando a um modelo hiperparametrizado, e com um desempenho superior para a aplicação.
                
    """)

    _model, X_test, pred, X_train, forecast_ = train_model._train_simple_prophet(df)
    baseline_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    baseline_r2 = round(r2_score(X_train['y'].values, forecast_['yhat'].values), 4)

with tab_resultados_iniciais:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{baseline_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{baseline_r2}**
                
    """)

with tab_conceitos:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da descoberta de uma sazonalidade de 365 dias, iremos utilizar alguns 
        conceitos mais avançados para melhorar o desempenho de nossa previsão:
                
        #### Validação Cruzada com Time Series Split
                
        Esta é uma técnica usada para avaliar o desempenho de modelos de aprendizado de máquina em dados de séries temporais. 
                
        A diferença para dessa técnica para a validação cruzada tradicional, onde os dados são embaralhados aleatoriamente, na validação cruzada com divisão de séries temporais, a ordem temporal dos dados é mantida, pois a dependência temporal é crucial em séries temporais.

        Sobre o processo:

        1. O 1º passo envolve a divisão do conjunto de dados em vários blocos ou dobras, onde cada bloco subsequente contém observações temporais mais recentes;
        2. Após isso, treinamos o modelo em cada conjunto de treinamento (treino)
        3. Na sequência avaliamos seu desempenho no conjunto de teste correspondente (teste). A métrica de avaliação é registrada para cada dobra.

    """)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(config.TIME_SERIES_SPLIT_IMG,
                caption="Time Series Split",
                width=300,  
        )

    st.markdown(f"""     
        Com isso, teremos várias métricas de desempenho para cada dobra. Isso nos ajudará a entender como o modelo se comporta em diferentes períodos da série temporal 
        e se ele consegue generalizar bem para dados futuros.
                

        #### Adição de feriados na modelagem do Prophet
                
        A adição de feriados na modelagem do Prophet (que já possui um mecanismo embutido para lidar esses eventos sazonais, como, por exemplo, os feriados) é uma técnica usada para melhorar a precisão das previsões em séries temporais ao levar em consideração esses eventos.

        Essa implementação permitirá o modelo ajustar seus componentes de sazonalidade e tendência conforme os padrões observados nos dados de feriados passados, resultando em previsões mais precisas em relação a esses eventos especiais.
    """)
    
    st.markdown(f"""            
        #### Hiperparametrização Bayesiana
                
        É uma abordagem para encontrar os melhores hiperparâmetros ( configurações que não são aprendidas diretamente pelo algoritmo durante o treinamento, 
        mas afetam como o modelo é treinado e como faz previsões) para um modelo de machine learning.

        Ela utiliza modelos probabilísticos e estatísticos para prever como diferentes configurações afetarão o desempenho do modelo.
        Em vez de tentar todas as combinações possíveis (o que pode ser computacionalmente caro), ela usa um modelo substituto para guiar a busca, 
        focando nas combinações mais promissoras e equilibrando a exploração do que não conhece com o aproveitamento do que já foi aprendido.        
    """)

with tab_variaveis_externas:
    st.markdown("""
        Após tais definições conceituais, definimos nossa função objetivo como:
    """)
    st.code("""            
        def objective(params):
            metrics, cv_mape = _run_cv_prophet(
                            df_model=X_train.dropna(),
                            params=params,
                            n_splits=5, test_size=test_size
                        )
            return {
                          'loss':cv_mape
                        , 'status': STATUS_OK
                    }
    """, language='python')  
    st.markdown("""          
        Ou seja, vamos minimizar o mape, definido como uma lista dos mapes dos 5 splits mencionados na função, 
        mantendo um test_size (ou, tamanho da base de teste) como 30 pontos (aproximadamente um mês);

        O espaço de busca do algoritmo vai ser definido como:
    """)
    st.code("""
        space = {
            'yearly_seasonality':365
            , 'daily_seasonality':hp.choice('daily_seasonality', [True, False])
            , 'weekly_seasonality':hp.choice('weekly_seasonality', [True, False])
            , 'seasonality_mode' : hp.choice('seasonality_mode', ['multiplicative','additive'])
            , "seasonality_prior_scale": hp.uniform("seasonality_prior_scale", 7, 10)
            , "changepoint_prior_scale": hp.uniform("changepoint_prior_scale", 0.4, 0.5)
            , "changepoint_range": hp.uniform("changepoint_range", 0.4, 0.5)
            , 'holidays_prior_scale' : hp.uniform('holidays_prior_scale', 7, 10)
            , "regressors":''
        }
    """, language='python')    
    st.markdown("""        
        Aqui, fixamos a sazonalidade anual como 365, mantendo a diária e semanal como o padrão do algoritmo,
        bem como deixamos o algoritmo de hiperparametrização definir qual a melhor configuração para os demais hiperparâmetros do modelo.
                
        Com tais configurações, chegamos ao seguinte melhor resultado:
    """)   

    st.code("best_params = "+str(train_model._get_best_params()).replace(",",",\n\t\t"), language='python')

    st.markdown("""        
        Para melhor visualizar o resultado da hiperparametrização, podemos verificar no seguinte gráfico, as áreas onde temos espaços mais
        "pretos", onde estão concentrados os resultados com menor erro percentual; Nota-se que existem vários "vales" de bons resultados, onde
        nossa hiperparametrização poderia ter retornado bons parâmetros;
        Para modificar o parâmetro sendo analisado, basta selecionar abaixo:
    """)   
    trials_df = get_data._trials()

    col1, col2 = st.columns(2)

    with col1:
        hyperparam_1 = st.selectbox(
            'Hiperparâmetro 1',
            list(set(trials_df.columns) - set(['loss']))
        )
    with col2:    
        hyperparam_2 = st.selectbox(
            'Hiperparâmetro 2',
            list(set(trials_df.columns) - set(['loss']))
        )

    st.plotly_chart(
        generate_graphs._plot_trials(trials_df, hyperparam_1, hyperparam_2)
    )

    _model, X_test, pred, X_train, forecast_ = train_model._train_cv_prophet(df)
    second_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    second_r2 = round(r2_score(X_train['y'].values, 
                          forecast_.loc[forecast_.ds.isin(X_train.ds.to_list())]['yhat'].values)
                , 4)
    sec_melhoria_mape = abs(round(second_mape - baseline_mape, 2))

with tab_deploy_producao:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{second_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{second_r2}**.

        Tais resultados, mostram uma melhoria de {sec_melhoria_mape}% em mape (considerando {baseline_mape}% como o anterior), em porcentagem absoluta!
                
    """)