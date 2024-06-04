import numpy as np
from joblib import load

# Esse script carrega os dois modelos treinados, e testa eles com uma instância do dataset.
# Lembrar que a saida do binario vai ser sempre 0 ou 1, (tem ou não a doença), já a saída do multiclasse vai ser um número de 0 a 5, representando a classe da doença.
# 0 - ASC-H 
# 1 - ASC-US 
# 2 - HSIL 
# 3 - LSIL 
# 4 - Negative for intraepithelial lesion 
# 5 - SCC 

# Carregar os modelos pré-treinados
model_binario = load('../Classificadores/ModelosTreinados/xgboostBinary_model.pkl')
model_multiclasse = load('../Classificadores/ModelosTreinados/xgboostMulti_model.pkl')

# Definir a entrada para ambos os modelos
entrada = '7.248304655095292,21.17803684411382,28.20973364173325,29.135240694328274,-59.46663875649005,-39.72426859380878,57.82618002775308,6.466844614986726,18.84047735308029,25.448004897625164,26.12834139099954,-53.525266476300445,35.628335712793984,51.93695413514918,4.4785120396360245,13.629832611574178,15.391241327616664,15.287393274888451,32.26078688725662,22.44070673768876,30.646120858935387,7.272455733398132,21.793772112872936,28.858041551777507,29.674772852086072,-60.84398169444117,-40.57291770094505,58.95242852493078,1'
entrada = np.array([float(x) for x in entrada.split(',')[:-1]]).reshape(1, -1)

# Fazer previsões utilizando o modelo binário
previsao_binario = model_binario.predict(entrada)
print(f'Previsão do modelo binário: {previsao_binario[0]}')

# Fazer previsões utilizando o modelo multi-classe
previsao_multiclasse = model_multiclasse.predict(entrada)
print(f'Previsão do modelo multi-classe: {previsao_multiclasse[0]}')
