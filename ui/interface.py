import cmd
import logging
from typing import Optional
from pathlib import Path
from ..core.network import SinNetwork
from ..models.model_manager import ModelManager
from ..core.level_system import LevelSystem

logger = logging.getLogger(__name__)

class CommandLineInterface(cmd.Cmd):
    def __init__(self, network: SinNetwork):
        super().__init__()
        self.sin = network
        self.model_manager = ModelManager()
        self.current_file = None
        self.plugins = {}  # Словарь загруженных плагинов

    def _load_plugins(self) -> Dict[str, SinPlugin]:
        plugins = {}
        # Динамическая загрузка плагинов
        plugins_dir = Path(__file__).parent.parent / "plugins"
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_") or plugin_file.name == "base.py":
                continue
            
            module_name = f"sin_ai.plugins.{plugin_file.stem}"
            try:
                module = __import__(module_name, fromlist=[''])
                for name, obj in module.__dict__.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, SinPlugin) and 
                        obj != SinPlugin):
                        plugin = obj()
                        plugin.initialize(self.sin)
                        plugins[plugin_file.stem] = plugin
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        return plugins
    
    def get_names(self):
        names = super().get_names()
        # Добавляем команды из плагинов
        for plugin in self.plugins.values():
            names.extend(f"do_{cmd}" for cmd in plugin.get_commands())
        return names
    
    def __getattr__(self, name):
        # Обработка команд плагинов
        if name.startswith('do_'):
            cmd = name[3:]
            for plugin in self.plugins.values():
                if cmd in plugin.get_commands():
                    return lambda arg: self._execute_plugin_command(plugin, cmd, arg)
        
        raise AttributeError(f"'CommandLineInterface' object has no attribute '{name}'")
    
    def _execute_plugin_command(self, plugin, command, args):
        result = plugin.execute_command(command, args)
        print(result)
    
    def do_plugins(self, arg):
        """Список загруженных плагинов и их команд"""
        if not self.plugins:
            print("Нет загруженных плагинов")
            return
        
        print("\nЗагруженные плагины:")
        for i, (name, plugin) in enumerate(self.plugins.items(), 1):
            print(f"{i}. {name}")
            for cmd, desc in plugin.get_commands().items():
                print(f"   {cmd}: {desc}")

     def register_plugins(self, plugins: Dict[str, SinPlugin]):
        """Регистрация плагинов в интерфейсе"""
        self.plugins = plugins
        for cmd_name, description in self.get_plugin_commands().items():
            setattr(self, f'do_{cmd_name}', self._create_plugin_command(cmd_name))

    def _create_plugin_command(self, cmd_name: str):
        """Динамическое создание методов для команд плагинов"""
        def plugin_command(args):
            for plugin in self.plugins.values():
                if cmd_name in plugin.get_commands():
                    result = plugin.execute_command(cmd_name, args)
                    print(result)
                    return
            print(f"Команда {cmd_name} не найдена в плагинах")
        return plugin_command

    def get_plugin_commands(self) -> Dict[str, str]:
        """Получение всех команд из плагинов"""
        commands = {}
        for plugin in self.plugins.values():
            commands.update(plugin.get_commands())
        return commands
    
    def precmd(self, line):
        logger.info(f"Command: {line}")
        return line
    
    def postcmd(self, stop, line):
        # После каждой команды показываем уровень и прогресс
        level_info = self.sin.level_system.get_level_info()
        print(f"\nLevel: {level_info['level']} | "
              f"EXP: {level_info['experience']}/{level_info['next_level_exp']} | "
              f"Progress: {level_info['progress']:.1%}\n")
        return stop
    
    def do_chat(self, arg):
        """Общение с Sin: chat [сообщение]"""
        if not arg:
            print("Пожалуйста, введите сообщение")
            return
        
        response = self.sin.communicate(arg)
        print(f"Sin: {response}")
    
    def do_learn(self, arg):
        """Обучение Sin из файла: learn [путь_к_файлу]"""
        if not arg:
            print("Пожалуйста, укажите путь к файлу")
            return
        
        file_path = Path(arg)
        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            return
        
        self.current_file = file_path
        print(f"Начало обучения из файла: {file_path.name}...")
        self.sin.learn_from_file(str(file_path))
        print("Обучение завершено")
    
    def do_learn_code(self, arg):
        """Обучение Sin на примере кода: learn_code [язык] [код или путь к файлу]"""
        args = arg.split(maxsplit=1)
        if len(args) < 2:
            print("Использование: learn_code [язык] [код или путь к файлу]")
            return
        
        language, code_or_path = args
        code_path = Path(code_or_path)
        
        if code_path.exists():
            with open(code_path, 'r') as f:
                code = f.read()
        else:
            code = code_or_path
        
        print(f"Начало обучения на {language} коде...")
        self.sin.learn_from_code(code, language)
        print("Обучение завершено")
    
    def do_query(self, arg):
        """Запрос к DeepSeek: query [запрос]"""
        if not arg:
            print("Пожалуйста, введите запрос")
            return
        
        print(f"Отправка запроса к DeepSeek: {arg}")
        response = self.sin.query_deepseek(arg)
        print(f"DeepSeek ответил:\n{response}")
    
    def do_save(self, arg):
        """Сохранение текущей модели: save [имя_модели]"""
        model_name = arg or f"sin_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Сохранение модели как {model_name}...")
        self.sin.save_model(model_name)
        print("Модель сохранена")
    
    def do_list(self, arg):
        """Список сохраненных моделей"""
        models = self.model_manager.list_models()
        if not models:
            print("Нет сохраненных моделей")
            return
        
        print("\nСохраненные модели:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['model_name']} ({model['save_date']})")
    
    def do_load(self, arg):
        """Загрузка модели: load [имя_модели или номер]"""
        if not arg:
            print("Пожалуйста, укажите имя модели или номер")
            return
        
        models = self.model_manager.list_models()
        if not models:
            print("Нет сохраненных моделей для загрузки")
            return
        
        # Попытка интерпретировать аргумент как номер
        try:
            model_num = int(arg)
            if 1 <= model_num <= len(models):
                model_name = models[model_num - 1]['model_name']
            else:
                print(f"Недопустимый номер модели. Допустимо от 1 до {len(models)}")
                return
        except ValueError:
            model_name = arg
        
        print(f"Загрузка модели {model_name}...")
        loaded_model = self.model_manager.load_model(model_name, SinNetwork)
        if loaded_model:
            self.sin = loaded_model
            print("Модель успешно загружена")
    
    def do_delete(self, arg):
        """Удаление модели: delete [имя_модели или номер]"""
        if not arg:
            print("Пожалуйста, укажите имя модели или номер")
            return
        
        models = self.model_manager.list_models()
        if not models:
            print("Нет сохраненных моделей для удаления")
            return
        
        # Попытка интерпретировать аргумент как номер
        try:
            model_num = int(arg)
            if 1 <= model_num <= len(models):
                model_name = models[model_num - 1]['model_name']
            else:
                print(f"Недопустимый номер модели. Допустимо от 1 до {len(models)}")
                return
        except ValueError:
            model_name = arg
        
        confirm = input(f"Вы уверены, что хотите удалить модель {model_name}? (y/n): ")
        if confirm.lower() == 'y':
            if self.model_manager.delete_model(model_name):
                print("Модель удалена")
            else:
                print("Не удалось удалить модель")
    
    def do_propose(self, arg):
        """Предложить новый функционал"""
        print("Sin анализирует свои возможности и предлагает улучшение...")
        feature = self.sin.propose_feature()
        
        print("\nПредлагаемый функционал:")
        print(f"Описание: {feature['description']}")
        print(f"Преимущества: {feature['benefits']}")
        
        if feature['code']:
            print("\nКод:")
            print(feature['code'])
        
        if feature['tests']:
            print("\nТесты:")
            print(feature['tests'])
        
        if feature['code'] and feature['tests']:
            confirm = input("\nПротестировать этот функционал? (y/n): ")
            if confirm.lower() == 'y':
                test_result = self.sin.test_feature(feature['code'], feature['tests'])
                self._display_test_result(test_result)
                
                if test_result.get('test_passed', False):
                    confirm_update = input("\nТесты пройдены. Обновить Sin? (y/n): ")
                    if confirm_update.lower() == 'y':
                        if self.sin.update_self(feature['code']):
                            print("Sin успешно обновлен!")
                        else:
                            print("Не удалось обновить Sin")
    
    def _display_test_result(self, test_result: Dict) -> None:
        """Отображение результатов тестирования"""
        print("\nРезультаты тестирования:")
        print(f"Функционал валиден: {'Да' if test_result['feature_valid'] else 'Нет'}")
        print(f"Тесты пройдены: {'Да' if test_result['test_passed'] else 'Нет'}")
        
        if test_result.get('feature_error'):
            print("\nОшибки в функционале:")
            print(test_result['feature_error'])
        
        if test_result.get('test_error'):
            print("\nОшибки в тестах:")
            print(test_result['test_error'])
        
        if test_result.get('test_output'):
            print("\nВывод тестов:")
            print(test_result['test_output'])
    
    def do_status(self, arg):
        """Показать статус обучения"""
        status = self.sin.get_learning_progress()
        level_info = self.sin.level_system.get_level_info()
        
        print("\nТекущий статус Sin:")
        print(f"Уровень: {level_info['level']}")
        print(f"Опыт: {level_info['experience']}/{level_info['next_level_exp']}")
        print(f"Прогресс до след. уровня: {level_info['progress']:.1%}")
        print(f"Всего сессий обучения: {level_info['total_sessions']}")
        print(f"В процессе обучения: {'Да' if status['is_learning'] else 'Нет'}")
        
        if status['recent_topics']:
            print("\nНедавние темы:")
            for i, topic in enumerate(status['recent_topics'], 1):
                print(f"{i}. {topic}")
    
    def do_exit(self, arg):
        """Выход из программы"""
        print("До свидания!")
        return True
    
    def do_EOF(self, arg):
        """Выход по Ctrl+D"""
        return self.do_exit(arg)
    
    def emptyline(self):
        pass
    
    def default(self, line):
        """Обработка нераспознанных команд как сообщений для чата"""
        return self.do_chat(line)
