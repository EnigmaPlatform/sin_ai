from sin_ai.core.network import SinNetwork
from sin_ai.ui.interface import CommandLineInterface
import logging

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sin_ai.log'),
            logging.StreamHandler()
        ]
    )

def main():
    configure_logging()
    
    # Инициализация нейросети
    sin = SinNetwork()
    
    # Запуск интерфейса
    cli = CommandLineInterface(sin)
    cli.run()

if __name__ == "__main__":
    main()
