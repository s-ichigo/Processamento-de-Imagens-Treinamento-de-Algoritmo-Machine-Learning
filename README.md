# Processamento-de-Imagens-Treinamento-de-Algoritmo-Machine-Learning

# Descrição do projeto:
Neste projeto, utilizamos o TeachbleMachine para criar um modelo de classificação de objetos. O TeachbleMachine é uma ferramenta do Google que permite criar modelos de aprendizado de máquina sem precisar escrever código inicialmente. Após definir as classes e coletar as imagens necessárias para cada classe usando a interface do TeachbleMachine, exportamos o modelo treinado como um arquivo TensorFlow (.h5).
Em seguida, em Python, carregamos esse modelo usando bibliotecas como TensorFlow  e aplica o modelo em um script para realizar inferências em novas imagens. O projeto também inclui a pré-processamento das imagens e a implementação de métricas para avaliar a precisão do modelo.
Este fluxo permite criar um sistema de classificação de objetos de forma rápida e eficiente, aproveitando a interface intuitiva do TeachbleMachine e a flexibilidade do Python.

# Instrução de instalação:
• Criar e Treinar o Modelo no TeachbleMachine
• Acessar TeachbleMachine.
• Escolhe uma opção de projeto "Image Project".
• Defina suas classes de objetos e faça o reconhecimentos das imagens representativas de cada classe.
• Treine o modelo diretamente na plataforma.
• Após o treinamento, clique em "Export Model" e selecione a opção de exportação para TensorFlow (.h5).
• Configuração do Ambiente Python
• Pré-requisitos: Instale o Python 3.9 em seu sistema.
• Criar um Ambiente Virtual .
• Configurar o Ambiente Python no Jupyter
• Criar um Novo Notebook Jupyter
• Carregar o Modelo e Fazer Inferências no Jupyter

#Instrução de uso:
• Configuração da Webcam Iriun
• Baixe e instale o aplicativo Iriun Webcam no celular (disponível para Android e iOS).
• No computador, instale a Webcam Iriun para o sistema operacional (Windows).
• Certifiquei de que o celular e o computador estivessem na mesma rede Wi-Fi.
• Abri o aplicativo Iriun no celular e no  Iriun no computador. Ele irá conectar automaticamente a webcam do celular no PC.
• Configuração do Ambiente Python no Jupyter
• Abrir e Criar um Novo Notebook no Jupyter
• Código para Capturar Imagens da Webcam e Fazer Inferências

# Este código captura imagens da webcam, processa-as e faz previsões usando o modelo treinado do TeachbleMachine:

    from keras.models import load_model  # TensorFlow is required for Keras to work
    import cv2  # Install opencv-python
    import numpy as np

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)

    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
        break

    camera.release()
    cv2.destroyAllWindows()


Créditos: Este projeto foi realizado por Marcilene Sarubi e Erison Lins como discentes da Universidade Federal do Oeste do Pará (UFOPA), com a contribuição do colaborador Tcharles Coutinho. O projeto foi desenvolvido e finalizado no Laboratório 2 (Lab2) da UFOPA, em 22 de outubro de 2024.
