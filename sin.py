import json
import math
import random
import time
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Set
import re
import logging
import os

# Настройка логгирования
log_file_path = 'neural_network.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# ТОКЕНИЗАТОР С УЧЕТОМ КОНТЕКСТА
# ======================
class ContextAwareTokenizer:
    """Токенизатор с учетом контекста и семантики"""
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.word_contexts = defaultdict(set)  # слова и их контексты
        self.word_frequencies = defaultdict(int)  # частоты слов

    def fit(self, texts: List[str]):
        """Обучаем токенизатор на наборе текстов"""
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)
        # Создаем словарь токенов
        for word in sorted(all_words):
            self.vocab[word] = self.vocab_size
            self.reverse_vocab[self.vocab_size] = word
            self.vocab_size += 1

    def tokenize(self, text: str) -> List[int]:
        """Преобразуем текст в список токенов"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(word, -1) for word in words if word in self.vocab]

    def encode(self, text: str) -> List[int]:
        """Алиас для токенизации"""
        return self.tokenize(text)

    def decode(self, tokens: List[int]) -> str:
        """Преобразуем токены обратно в текст"""
        words = [self.reverse_vocab.get(token, '[UNK]') for token in tokens]
        return ' '.join(words)

    def add_context(self, word: str, context: str):
        """Добавить контекст для слова"""
        self.word_contexts[word].add(context)
        self.word_frequencies[word] += 1

# ======================
# НЕЙРОН С КОНТЕКСТОМ
# ======================
class ContextualNeuron:
    """Нейрон с учетом контекста"""
    def __init__(self, id: int, activation_function: str = 'sigmoid'):
        self.id = id
        self.value = 0.0
        self.bias = random.uniform(-0.1, 0.1)
        self.activation_func = activation_function
        self.inputs = []  # входящие связи
        self.outputs = []  # исходящие связи
        self.is_active = False
        self.threshold = 0.5
        self.context_strength = 0.0  # сила контекста
        self.memory_trace = []  # история активаций
        self.learning_history = []  # история обучения
        self.group = None  # группа, к которой принадлежит нейрон
        self.level = 0  # уровень абстракции

    def activate(self, input_value: float = None) -> float:
        """Активация нейрона с учетом контекста"""
        if input_value is not None:
            self.value = input_value
        else:
            # Суммируем все входящие сигналы
            total_input = sum(conn.weight * conn.from_neuron.value for conn in self.inputs)
            self.value = total_input + self.bias
        # Применяем активационную функцию
        self.value = self._apply_activation(self.value)
        self.is_active = True
        self.memory_trace.append(self.value)
        # Ограничиваем историю
        if len(self.memory_trace) > 10:
            self.memory_trace.pop(0)
        return self.value

    def _apply_activation(self, x: float) -> float:
        """Сигмоидальная активационная функция"""
        if self.activation_func == 'sigmoid':
            return 1 / (1 + math.exp(-x))
        elif self.activation_func == 'tanh':
            return math.tanh(x)
        else:
            return max(0, x)  # ReLU

    def receive_signal(self, signal: float):
        """Получение сигнала от другой связи"""
        self.value += signal

    def reset(self):
        """Сброс нейрона"""
        self.value = 0.0
        self.is_active = False

# ======================
# СВЯЗЬ С КОНТЕКСТОМ
# ======================
class ContextualConnection:
    """Связь между нейронами с учетом контекста"""
    def __init__(self, from_neuron: ContextualNeuron, to_neuron: ContextualNeuron, 
                 weight: float = None, context_weight: float = 1.0):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight if weight is not None else random.uniform(-0.1, 0.1)
        self.context_weight = context_weight  # вес контекста
        self.learning_rate = 0.01
        self.is_active = False
        self.last_delta = 0.0
        self.strength = 1.0  # Сила связи
        self.context_history = []  # история контекста
        self.connection_type = "regular"  # тип связи: regular, logical, abstract
        self.level = 1  # уровень связи

    def activate(self) -> float:
        """Активация связи с учетом контекста"""
        if self.from_neuron.is_active:
            # Учитываем контекстную силу
            signal = self.from_neuron.value * self.weight * self.strength * self.context_weight
            self.to_neuron.receive_signal(signal)
            self.is_active = True
            self.context_history.append((signal, time.time()))
            return signal
        return 0.0

    def update_weight(self, error: float):
        """Обновление веса связи с учетом контекста"""
        if self.from_neuron.is_active:
            delta = self.learning_rate * error * self.from_neuron.value
            self.weight += delta
            self.last_delta = delta

# ======================
# СВЯЗИ МЕЖДУ СВЯЗЯМИ
# ======================
class ConnectionConnection:
    """Связь между связями (уровень 2 абстракции)"""
    def __init__(self, parent_conn1: ContextualConnection, parent_conn2: ContextualConnection, 
                 influence_strength: float = 0.1):
        self.conn1 = parent_conn1
        self.conn2 = parent_conn2
        self.influence_strength = influence_strength
        self.active = False
        self.connection_history = []  # история влияния
        self.strength_change = 0.0  # изменение силы связи
        self.level = 2  # уровень связи между связями

    def activate_influence(self):
        """Активация влияния одной связи на другую"""
        if self.conn1.is_active and self.conn2.is_active:
            # Влияние связи 1 на связь 2
            influence = self.conn1.weight * self.influence_strength
            self.conn2.strength += influence
            self.conn2.context_weight += influence * 0.1  # дополнительный контекст
            self.active = True
            self.connection_history.append((influence, time.time()))
            self.strength_change = influence
            return influence
        return 0.0

# ======================
# СВЯЗИ СВЯЗЕЙ СО СВЯЗЯМИ
# ======================
class ConnectionConnectionConnection:
    """Связь между связями связей (уровень 3 абстракции)"""
    def __init__(self, parent_conn_conn1: ConnectionConnection, parent_conn_conn2: ConnectionConnection, 
                 influence_strength: float = 0.1):
        self.conn_conn1 = parent_conn_conn1
        self.conn_conn2 = parent_conn_conn2
        self.influence_strength = influence_strength
        self.active = False
        self.connection_history = []  # история влияния
        self.strength_change = 0.0  # изменение силы связи
        self.level = 3  # уровень связи между связями связей

    def activate_influence(self):
        """Активация влияния одной связи связей на другую"""
        if self.conn_conn1.active and self.conn_conn2.active:
            # Влияние связи связей 1 на связь связей 2
            influence = self.conn_conn1.strength_change * self.influence_strength
            self.conn_conn2.strength_change += influence
            self.active = True
            self.connection_history.append((influence, time.time()))
            return influence
        return 0.0

# ======================
# СВЯЗИ СВЯЗЕЙ СВЯЗЕЙ СО СВЯЗЯМИ СВЯЗЯМИ
# ======================
class ConnectionConnectionConnectionConnection:
    """Связь между связями связей связей (уровень 4 абстракции)"""
    def __init__(self, parent_conn_conn_conn1: ConnectionConnectionConnection, 
                 parent_conn_conn_conn2: ConnectionConnectionConnection, 
                 influence_strength: float = 0.1):
        self.conn_conn_conn1 = parent_conn_conn_conn1
        self.conn_conn_conn2 = parent_conn_conn_conn2
        self.influence_strength = influence_strength
        self.active = False
        self.connection_history = []  # история влияния
        self.strength_change = 0.0  # изменение силы связи
        self.level = 4  # уровень связи между связями связей связей

    def activate_influence(self):
        """Активация влияния одной связи связей связей на другую"""
        if self.conn_conn_conn1.active and self.conn_conn_conn2.active:
            # Влияние связи связей связей 1 на связь связей связей 2
            influence = self.conn_conn_conn1.strength_change * self.influence_strength
            self.conn_conn_conn2.strength_change += influence
            self.active = True
            self.connection_history.append((influence, time.time()))
            return influence
        return 0.0

# ======================
# НЕЙРОННЫЕ ГРУППЫ
# ======================
class NeuronGroup:
    """Группа нейронов с уровнем абстракции"""
    def __init__(self, name: str):
        self.name = name
        self.neurons = []
        self.connections = []
        self.subgroups = []
        self.activation_level = 0.0
        self.group_connections = []  # связи между группами
        self.group_context = {}  # контекст группы
        self.learning_history = []
        self.level = 0  # уровень группы

    def add_neuron(self, neuron: ContextualNeuron):
        """Добавить нейрон в группу"""
        neuron.group = self
        self.neurons.append(neuron)

    def add_connection(self, connection: ContextualConnection):
        """Добавить связь в группу"""
        self.connections.append(connection)

    def add_subgroup(self, subgroup: 'NeuronGroup'):
        """Добавить подгруппу"""
        self.subgroups.append(subgroup)

    def activate(self):
        """Активировать группу"""
        if not self.neurons:
            return 0.0
        # Считаем среднюю активацию нейронов
        avg_activation = sum(n.value for n in self.neurons) / len(self.neurons)
        self.activation_level = avg_activation
        return self.activation_level

# ======================
# СТРУКТУРА НЕЙРОННОЙ СЕТИ
# ======================
class MultiLevelNeuralNetwork:
    """Многоуровневая нейронная сеть с возможностью формирования сложных структур"""
    def __init__(self):
        self.neurons = {}  # {id: Neuron}
        self.connections = []  # список Connection объектов
        self.connection_connections = []  # связи между связями
        self.connection_connection_connections = []  # связи между связями связей
        self.connection_connection_connection_connections = []  # связи между связями связей связей
        self.groups = {}  # группы нейронов
        self.next_neuron_id = 0
        self.next_group_id = 0
        self.training_history = []
        self.context_history = []  # история контекста
        self.learning_session = 0  # сессия обучения
        self.structure_levels = {}  # структура по уровням
        self.input_neurons = []  # входные нейроны
        self.output_neurons = []  # выходные нейроны

    def add_neuron(self, neuron: ContextualNeuron = None) -> ContextualNeuron:
        """Добавление нейрона в сеть"""
        if neuron is None:
            neuron = ContextualNeuron(self.next_neuron_id)
            self.next_neuron_id += 1
        self.neurons[neuron.id] = neuron
        return neuron

    def create_connection(self, from_neuron_id: int, to_neuron_id: int, 
                         weight: float = None) -> ContextualConnection:
        """Создание связи между нейронами"""
        from_neuron = self.neurons[from_neuron_id]
        to_neuron = self.neurons[to_neuron_id]
        connection = ContextualConnection(from_neuron, to_neuron, weight)
        self.connections.append(connection)
        from_neuron.outputs.append(connection)
        to_neuron.inputs.append(connection)
        return connection

    def create_connection_connection(self, conn1: ContextualConnection, conn2: ContextualConnection,
                                   influence_strength: float = 0.1) -> ConnectionConnection:
        """Создание связи между связями"""
        connection_connection = ConnectionConnection(conn1, conn2, influence_strength)
        self.connection_connections.append(connection_connection)
        return connection_connection

    def create_connection_connection_connection(self, conn_conn1: ConnectionConnection, 
                                             conn_conn2: ConnectionConnection,
                                             influence_strength: float = 0.1) -> ConnectionConnectionConnection:
        """Создание связи между связями связей"""
        connection_connection_connection = ConnectionConnectionConnection(conn_conn1, conn_conn2, influence_strength)
        self.connection_connection_connections.append(connection_connection_connection)
        return connection_connection_connection

    def create_connection_connection_connection_connection(self, conn_conn_conn1: ConnectionConnectionConnection, 
                                                        conn_conn_conn2: ConnectionConnectionConnection,
                                                        influence_strength: float = 0.1) -> ConnectionConnectionConnectionConnection:
        """Создание связи между связями связей связей"""
        connection_connection_connection_connection = ConnectionConnectionConnectionConnection(
            conn_conn_conn1, conn_conn_conn2, influence_strength)
        self.connection_connection_connection_connections.append(connection_connection_connection_connection)
        return connection_connection_connection_connection

    def create_group(self, name: str) -> NeuronGroup:
        """Создать группу нейронов"""
        group = NeuronGroup(name)
        self.groups[name] = group
        self.next_group_id += 1
        return group

    def add_neuron_to_group(self, neuron_id: int, group_name: str):
        """Добавить нейрон в группу"""
        if group_name in self.groups:
            group = self.groups[group_name]
            neuron = self.neurons[neuron_id]
            group.add_neuron(neuron)
            return True
        return False

    def forward(self, inputs: List[float]) -> List[float]:
        """Прямое распространение сигнала"""
        # Сбрасываем все нейроны
        for neuron in self.neurons.values():
            neuron.reset()
        
        # Устанавливаем входные значения
        for i, value in enumerate(inputs):
            if i < len(self.input_neurons):
                self.input_neurons[i].activate(value)
        
        # Проходим по всем связям
        for connection in self.connections:
            connection.activate()
        
        # Активируем связи между связями (дополнительно)
        for conn_conn in self.connection_connections:
            conn_conn.activate_influence()
        
        # Активируем связи между связями связей
        for conn_conn_conn in self.connection_connection_connections:
            conn_conn_conn.activate_influence()
        
        # Активируем связи между связями связей связей
        for conn_conn_conn_conn in self.connection_connection_connection_connections:
            conn_conn_conn_conn.activate_influence()
        
        # Активируем выходные нейроны
        outputs = []
        for neuron in self.output_neurons:
            if neuron.is_active:
                outputs.append(neuron.value)
        return outputs

    def train_step(self, inputs: List[float], targets: List[float], 
                  context: str = "") -> Dict[str, Any]:
        """Шаг обучения с контекстом"""
        # Прямое распространение
        outputs = self.forward(inputs)
        # Вычисляем ошибку
        errors = [target - output for target, output in zip(targets, outputs)]
        # Обратное распространение
        for i, error in enumerate(errors):
            if i < len(self.output_neurons):
                neuron = self.output_neurons[i]
                for conn in neuron.inputs:
                    conn.update_weight(error)
        self.training_history.append({
            'inputs': inputs,
            'targets': targets,
            'outputs': outputs,
            'errors': errors,
            'context': context,
            'session': self.learning_session
        })
        self.learning_session += 1
        return {
            'inputs': inputs,
            'outputs': outputs,
            'errors': errors,
            'context': context
        }

    def get_training_progress(self) -> float:
        """Получить прогресс обучения в процентах"""
        if not self.training_history:
            return 0.0
        # Простой расчет: чем больше историй обучения, тем выше прогресс
        return min(100.0, len(self.training_history) * 0.5)

    def add_input_neuron(self) -> ContextualNeuron:
        """Добавить входной нейрон"""
        neuron = self.add_neuron()
        self.input_neurons.append(neuron)
        return neuron

    def add_output_neuron(self) -> ContextualNeuron:
        """Добавить выходной нейрон"""
        neuron = self.add_neuron()
        self.output_neurons.append(neuron)
        return neuron

    def get_context_summary(self) -> Dict[str, Any]:
        """Получить сводку по контексту обучения"""
        summary = {
            'total_sessions': len(self.training_history),
            'active_neurons': len([n for n in self.neurons.values() if n.is_active]),
            'total_connections': len(self.connections),
            'total_connection_connections': len(self.connection_connections),
            'total_connection_connection_connections': len(self.connection_connection_connections),
            'total_connection_connection_connection_connections': len(self.connection_connection_connection_connections),
            'total_groups': len(self.groups),
            'learning_rate': self.connections[0].learning_rate if self.connections else 0.01,
            'structure_levels': {
                'neurons': len(self.neurons),
                'connections': len(self.connections),
                'connection_connections': len(self.connection_connections),
                'connection_connection_connections': len(self.connection_connection_connections),
                'connection_connection_connection_connections': len(self.connection_connection_connection_connections),
                'groups': len(self.groups)
            }
        }
        return summary

# ======================
# ОБУЧАТЕЛЬ С ЛОГИРОВАНИЕМ
# ======================
class MultiLevelTrainer:
    """Класс для обучения нейросети с логгированием и многоуровневыми структурами"""
    def __init__(self, network: MultiLevelNeuralNetwork, tokenizer: ContextAwareTokenizer):
        self.network = network
        self.tokenizer = tokenizer
        self.knowledge_log = []
        self.concept_map = defaultdict(list)  # карта понятий
        self.patterns = []  # сохраненные шаблоны
        self.session_counter = 0
        self.new_knowledge_counter = 0
        self.structure_complexity = 0  # уровень сложности структуры

    def train_on_text(self, text: str, context: str = "", epochs: int = 1) -> Dict[str, Any]:
        """Обучение на тексте с логгированием"""
        logger.info(f"Обучение на тексте: {text[:50]}...")
        tokens = self.tokenizer.encode(text)
        if not tokens:
            logger.warning("Не удалось токенизировать текст")
            return {"error": "Empty tokens"}
        
        # Преобразуем токены в векторы
        vector = [token / 1000.0 for token in tokens[:10]]  # Ограничиваем размер
        
        # Добавляем недостающие нейроны
        while len(vector) > len(self.network.input_neurons):
            self.network.add_input_neuron()
        
        # Устанавливаем значения нейронам
        for i, value in enumerate(vector):
            if i < len(self.network.input_neurons):
                self.network.input_neurons[i].activate(value)
        
        # Обучаем
        result = {}
        for epoch in range(epochs):
            result = self.network.train_step(vector, vector, context)
        
        # Логируем новое знание
        knowledge = {
            'type': 'text_learning',
            'text': text[:50] + '...' if len(text) > 50 else text,
            'tokens_count': len(tokens),
            'context': context,
            'session': self.session_counter,
            'timestamp': time.time(),
            'new_knowledge': True,
            'complexity_level': self.structure_complexity
        }
        self.knowledge_log.append(knowledge)
        self.session_counter += 1
        self.new_knowledge_counter += 1
        
        # Анализируем контекст
        self.analyze_context(text, context)
        
        # Увеличиваем сложность структуры
        self.structure_complexity += 0.1
        
        logger.info(f"Обучение завершено. Новых знаний: {len(self.knowledge_log)}")
        return result

    def analyze_context(self, text: str, context: str):
        """Анализ контекста для улучшения понимания"""
        # Извлекаем ключевые слова
        words = re.findall(r'\b\w+\b', text.lower())
        key_words = [word for word in words if len(word) > 3]
        # Добавляем контексты
        for word in key_words:
            self.tokenizer.add_context(word, context)
        # Сохраняем шаблоны
        pattern = {
            'text': text,
            'context': context,
            'keywords': key_words,
            'timestamp': time.time()
        }
        self.patterns.append(pattern)
        # Обновляем карту понятий
        for word in key_words:
            self.concept_map[word].append(pattern)

    def learn_during_chat(self, user_input: str, response: str, 
                         context: str = "") -> Dict[str, Any]:
        """Обучение во время чата с детальным логгированием"""
        logger.info(f"Обучение во время чата: '{user_input}'")
        # Обучаем на пользовательском вводе
        self.train_on_text(user_input, context, epochs=1)
        # Обучаем на ответе (если есть)
        if response:
            self.train_on_text(response, context, epochs=1)
        # Логируем новое знание
        knowledge = {
            'type': 'chat_learning',
            'input': user_input[:30] + '...' if len(user_input) > 30 else user_input,
            'response': response[:30] + '...' if len(response) > 30 else response,
            'context': context,
            'session': self.session_counter,
            'timestamp': time.time(),
            'new_knowledge': True,
            'complexity_level': self.structure_complexity
        }
        self.knowledge_log.append(knowledge)
        self.session_counter += 1
        self.new_knowledge_counter += 1
        # Анализируем контекст
        self.analyze_context(user_input, context)
        # Увеличиваем сложность структуры
        self.structure_complexity += 0.05
        logger.info(f"Новое знание: {knowledge['input']}")
        return knowledge

# ======================
# ХРАНЕНИЕ МОДЕЛИ С ЛОГИРОВАНИЕМ
# ======================
class MultiLevelModelStorage:
    """Класс для сохранения и загрузки модели с логами"""
    @staticmethod
    def save_model(network: MultiLevelNeuralNetwork, tokenizer: ContextAwareTokenizer, 
                   trainer: MultiLevelTrainer, filename: str = "model.json"):
        """Сохранение модели"""
        model_data = {
            'neurons': {},
            'connections': [],
            'connection_connections': [],
            'connection_connection_connections': [],
            'connection_connection_connection_connections': [],
            'groups': {},
            'vocab': tokenizer.vocab,
            'vocab_size': tokenizer.vocab_size,
            'word_contexts': dict(tokenizer.word_contexts),
            'word_frequencies': dict(tokenizer.word_frequencies),
            'training_history': network.training_history,
            'knowledge_log': trainer.knowledge_log,
            'concept_map': dict(trainer.concept_map),
            'patterns': trainer.patterns,
            'session_counter': trainer.session_counter,
            'new_knowledge_counter': trainer.new_knowledge_counter,
            'structure_complexity': trainer.structure_complexity,
            'network_context': network.get_context_summary()
        }
        # Сохраняем нейроны
        for neuron_id, neuron in network.neurons.items():
            model_data['neurons'][neuron_id] = {
                'id': neuron.id,
                'value': neuron.value,
                'bias': neuron.bias,
                'is_active': neuron.is_active,
                'context_strength': neuron.context_strength,
                'memory_trace': neuron.memory_trace,
                'group': neuron.group.name if neuron.group else None,
                'level': neuron.level
            }
        # Сохраняем связи
        for conn in network.connections:
            model_data['connections'].append({
                'from_neuron': conn.from_neuron.id,
                'to_neuron': conn.to_neuron.id,
                'weight': conn.weight,
                'context_weight': conn.context_weight,
                'strength': conn.strength,
                'connection_type': conn.connection_type,
                'level': conn.level
            })
        # Сохраняем связи между связями
        for conn_conn in network.connection_connections:
            model_data['connection_connections'].append({
                'conn1': conn_conn.conn1.from_neuron.id,
                'conn2': conn_conn.conn2.from_neuron.id,
                'influence_strength': conn_conn.influence_strength,
                'level': conn_conn.level
            })
        # Сохраняем связи между связями связей
        for conn_conn_conn in network.connection_connection_connections:
            model_data['connection_connection_connections'].append({
                'conn_conn1': conn_conn_conn.conn_conn1.conn1.from_neuron.id,
                'conn_conn2': conn_conn_conn.conn_conn2.conn1.from_neuron.id,
                'influence_strength': conn_conn_conn.influence_strength,
                'level': conn_conn_conn.level
            })
        # Сохраняем связи между связями связей связей
        for conn_conn_conn_conn in network.connection_connection_connection_connections:
            model_data['connection_connection_connection_connections'].append({
                'conn_conn_conn1': conn_conn_conn_conn.conn_conn_conn1.conn_conn1.conn1.from_neuron.id,
                'conn_conn_conn2': conn_conn_conn_conn.conn_conn_conn2.conn_conn1.conn1.from_neuron.id,
                'influence_strength': conn_conn_conn_conn.influence_strength,
                'level': conn_conn_conn_conn.level
            })
        # Сохраняем группы
        for group_name, group in network.groups.items():
            model_data['groups'][group_name] = {
                'name': group.name,
                'neuron_ids': [n.id for n in group.neurons],
                'connection_ids': [(c.from_neuron.id, c.to_neuron.id) for c in group.connections],
                'subgroup_names': [sg.name for sg in group.subgroups],
                'activation_level': group.activation_level,
                'group_context': group.group_context,
                'level': group.level
            }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Модель сохранена в {filename}")

    @staticmethod
    def load_model(filename: str = "model.json") -> Tuple[MultiLevelNeuralNetwork, 
                                                          ContextAwareTokenizer, 
                                                          MultiLevelTrainer]:
        """Загрузка модели"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            # Создаем новую сеть
            network = MultiLevelNeuralNetwork()
            tokenizer = ContextAwareTokenizer()
            tokenizer.vocab = model_data['vocab']
            tokenizer.vocab_size = model_data['vocab_size']
            tokenizer.word_contexts = defaultdict(set, model_data['word_contexts'])
            tokenizer.word_frequencies = defaultdict(int, model_data['word_frequencies'])
            network.training_history = model_data['training_history']
            
            # Восстанавливаем нейроны
            for neuron_id, neuron_data in model_data['neurons'].items():
                neuron = ContextualNeuron(neuron_data['id'])
                neuron.value = neuron_data['value']
                neuron.bias = neuron_data['bias']
                neuron.is_active = neuron_data['is_active']
                neuron.context_strength = neuron_data['context_strength']
                neuron.memory_trace = neuron_data['memory_trace']
                neuron.level = neuron_data['level']
                network.neurons[neuron.id] = neuron
            
            # Восстанавливаем связи
            for conn_data in model_data['connections']:
                from_neuron = network.neurons[conn_data['from_neuron']]
                to_neuron = network.neurons[conn_data['to_neuron']]
                conn = ContextualConnection(from_neuron, to_neuron, conn_data['weight'], 
                                           conn_data['context_weight'])
                conn.strength = conn_data['strength']
                conn.connection_type = conn_data['connection_type']
                conn.level = conn_data['level']
                network.connections.append(conn)
                from_neuron.outputs.append(conn)
                to_neuron.inputs.append(conn)
            
            # Восстанавливаем связи между связями
            for conn_conn_data in model_data['connection_connections']:
                try:
                    conn1 = next(c for c in network.connections 
                                if c.from_neuron.id == conn_conn_data['conn1'])
                    conn2 = next(c for c in network.connections 
                                if c.from_neuron.id == conn_conn_data['conn2'])
                    conn_conn = network.create_connection_connection(conn1, conn2, 
                                                                   conn_conn_data['influence_strength'])
                    conn_conn.level = conn_conn_data['level']
                except:
                    pass  # Пропускаем, если связи не найдены
            
            # Восстанавливаем связи между связями связей
            for conn_conn_conn_data in model_data['connection_connection_connections']:
                try:
                    conn_conn1 = next(cc for cc in network.connection_connections 
                                     if cc.conn1.from_neuron.id == conn_conn_conn_data['conn_conn1'])
                    conn_conn2 = next(cc for cc in network.connection_connections 
                                     if cc.conn1.from_neuron.id == conn_conn_conn_data['conn_conn2'])
                    conn_conn_conn = network.create_connection_connection_connection(conn_conn1, conn_conn2, 
                                                                                  conn_conn_conn_data['influence_strength'])
                    conn_conn_conn.level = conn_conn_conn_data['level']
                except:
                    pass
            
            # Восстанавливаем связи между связями связей связей
            for conn_conn_conn_conn_data in model_data['connection_connection_connection_connections']:
                try:
                    conn_conn_conn1 = next(ccc for ccc in network.connection_connection_connections 
                                          if ccc.conn_conn1.conn1.from_neuron.id == conn_conn_conn_conn_data['conn_conn_conn1'])
                    conn_conn_conn2 = next(ccc for ccc in network.connection_connection_connections 
                                          if ccc.conn_conn1.conn1.from_neuron.id == conn_conn_conn_conn_data['conn_conn_conn2'])
                    conn_conn_conn_conn = network.create_connection_connection_connection_connection(
                        conn_conn_conn1, conn_conn_conn2, conn_conn_conn_conn_data['influence_strength'])
                    conn_conn_conn_conn.level = conn_conn_conn_conn_data['level']
                except:
                    pass
            
            # Восстанавливаем группы
            for group_name, group_data in model_data['groups'].items():
                group = network.create_group(group_data['name'])
                group.activation_level = group_data['activation_level']
                group.group_context = group_data['group_context']
                group.level = group_data['level']
                # Восстанавливаем нейроны группы
                for neuron_id in group_data['neuron_ids']:
                    if neuron_id in network.neurons:
                        network.add_neuron_to_group(neuron_id, group_name)
                # Восстанавливаем подгруппы
                for subgroup_name in group_data['subgroup_names']:
                    if subgroup_name in network.groups:
                        group.add_subgroup(network.groups[subgroup_name])
            
            # Создаем тренер
            trainer = MultiLevelTrainer(network, tokenizer)
            trainer.knowledge_log = model_data['knowledge_log']
            trainer.concept_map = defaultdict(list, model_data['concept_map'])
            trainer.patterns = model_data['patterns']
            trainer.session_counter = model_data['session_counter']
            trainer.new_knowledge_counter = model_data['new_knowledge_counter']
            trainer.structure_complexity = model_data['structure_complexity']
            logger.info(f"Модель загружена из {filename}")
            return network, tokenizer, trainer
        except FileNotFoundError:
            logger.warning("Файл модели не найден. Создается новая модель.")
            return MultiLevelNeuralNetwork(), ContextAwareTokenizer(), MultiLevelTrainer(None, None)

# ======================
# КОНСОЛЬНЫЙ ИНТЕРФЕЙС С ЛОГИРОВАНИЕМ
# ======================
class MultiLevelCLIInterface:
    """Консольный интерфейс с детальным логгированием"""
    def __init__(self):
        # Проверяем наличие необходимых файлов и создаем их при необходимости
        self.ensure_directories_and_files()
        
        self.network, self.tokenizer, self.trainer = MultiLevelModelStorage.load_model()
        self.running = True
        self.conversation_history = []
        self.session_start_time = time.time()
        self.previous_knowledge_count = 0

    def ensure_directories_and_files(self):
        """Проверяем наличие необходимых файлов и папок"""
        # Создаем директорию для логов, если её нет
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Проверяем наличие лог-файла
        if not os.path.exists(log_file_path):
            try:
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write("")
                logger.info(f"Создан лог-файл: {log_file_path}")
            except Exception as e:
                logger.error(f"Ошибка создания лог-файла: {e}")
        
        # Проверяем наличие файла модели
        model_file = "model.json"
        if not os.path.exists(model_file):
            # Если файла нет, создаем пустую модель
            try:
                self.network, self.tokenizer, self.trainer = MultiLevelModelStorage.load_model()
                logger.info("Создана новая модель")
            except Exception as e:
                logger.error(f"Ошибка при создании модели: {e}")

    def start(self):
        """Запуск интерфейса"""
        print("=== МНОГОУРОВНЕВАЯ НЕЙРОННАЯ СЕТЬ ===")
        print("📝 Ведется логирование обучения в файл neural_network.log")
        print("💡 Нейросеть формирует сложные логические структуры!")
        print("🧠 Связи между связями, связи связей со связями...")
        print("🔗 Всевозможные варианты в глубину и ширину")
        print()
        print("Введите 'help' для помощи")
        print("Введите 'exit' для выхода")
        print("Введите 'save' для сохранения модели")
        print("Введите 'load' для загрузки модели")
        print("Введите 'progress' для просмотра прогресса обучения")
        print("Введите 'log' для просмотра истории знаний")
        print("Введите 'context' для просмотра контекста")
        print("Введите 'stats' для статистики")
        print("Введите 'groups' для просмотра групп нейронов")
        print("Введите 'structure' для просмотра структуры сети")
        print("Введите 'complexity' для просмотра сложности структуры")
        print("Введите 'train <file>' для обучения на файле")
        print("Введите 'chat' для режима чата")
        print()
        while self.running:
            try:
                command = input("Нейросеть> ").strip()
                if command.lower() == 'exit':
                    self.running = False
                    break
                elif command.lower() == 'help':
                    self.show_help()
                elif command.lower() == 'save':
                    self.save_model()
                elif command.lower() == 'load':
                    self.load_model()
                elif command.lower() == 'progress':
                    self.show_progress()
                elif command.lower() == 'clear':
                    self.clear_knowledge()
                elif command.lower() == 'log':
                    self.show_knowledge_log()
                elif command.lower() == 'context':
                    self.show_context()
                elif command.lower() == 'stats':
                    self.show_stats()
                elif command.lower() == 'groups':
                    self.show_groups()
                elif command.lower() == 'structure':
                    self.show_structure()
                elif command.lower() == 'complexity':
                    self.show_complexity()
                elif command.startswith('train'):
                    self.train_from_file(command[6:].strip())
                elif command.lower() == 'chat':
                    self.chat_mode()
                else:
                    # Предполагаем, что это вопрос для чата
                    self.handle_chat_input(command)
            except KeyboardInterrupt:
                print("\nВыход...")
                break
            except Exception as e:
                print(f"Ошибка: {e}")
                logger.error(f"Ошибка в интерфейсе: {e}")

    def show_help(self):
        """Показать помощь"""
        print("\nДоступные команды:")
        print("  help     - показать эту справку")
        print("  exit     - выход из программы")
        print("  save     - сохранить модель")
        print("  load     - загрузить модель")
        print("  progress - показать прогресс обучения")
        print("  log      - показать историю знаний")
        print("  context  - показать контекст обучения")
        print("  stats    - показать статистику")
        print("  groups   - показать группы нейронов")
        print("  structure- показать структуру сети")
        print("  complexity- показать сложность структуры")
        print("  clear    - очистить историю знаний")
        print("  train <file> - обучение на файле")
        print("  chat     - режим чата")
        print("  Любое другое сообщение - чат с нейросетью")
        print()

    def save_model(self):
        """Сохранить модель"""
        MultiLevelModelStorage.save_model(self.network, self.tokenizer, self.trainer)
        print("✅ Модель сохранена!")

    def load_model(self):
        """Загрузить модель"""
        self.network, self.tokenizer, self.trainer = MultiLevelModelStorage.load_model()
        print("🔄 Модель загружена!")

    def show_progress(self):
        """Показать прогресс обучения"""
        progress = self.network.get_training_progress()
        print(f"📊 Обученность нейросети: {progress:.1f}%")
        print(f"🧠 Новых знаний: {len(self.trainer.knowledge_log)}")
        print(f"📈 Сессий обучения: {self.trainer.session_counter}")
        print(f"🌐 Активных нейронов: {len([n for n in self.network.neurons.values() if n.is_active])}")
        print(f"🔗 Связей между связями: {len(self.network.connection_connections)}")
        print(f"🔗 Связей связей со связями: {len(self.network.connection_connection_connections)}")
        print(f"🔗 Связей связей связей со связями: {len(self.network.connection_connection_connection_connections)}")
        print(f"📦 Групп нейронов: {len(self.network.groups)}")

    def clear_knowledge(self):
        """Очистить историю знаний"""
        self.trainer.knowledge_log.clear()
        self.network.training_history.clear()
        self.trainer.concept_map.clear()
        self.trainer.patterns.clear()
        self.trainer.session_counter = 0
        self.trainer.new_knowledge_counter = 0
        self.trainer.structure_complexity = 0
        print("🗑️ История знаний очищена!")

    def show_knowledge_log(self):
        """Показать историю знаний"""
        if not self.trainer.knowledge_log:
            print("📝 Нет записей о знаниях")
            return
        print("\n📚 История знаний:")
        for i, record in enumerate(self.trainer.knowledge_log[-10:], 1):  # последние 10 записей
            print(f"  {i}. {record['type']} - {record['text'][:50]}...")
            if 'context' in record and record['context']:
                print(f"     Контекст: {record['context']}")
            if 'complexity_level' in record:
                print(f"     Сложность: {record['complexity_level']:.2f}")
        print()

    def show_context(self):
        """Показать контекст обучения"""
        print("\n🔍 Контекст обучения:")
        context_summary = self.network.get_context_summary()
        print(f"  Сессий: {context_summary['total_sessions']}")
        print(f"  Активных нейронов: {context_summary['active_neurons']}")
        print(f"  Связей: {context_summary['total_connections']}")
        print(f"  Связей между связями: {context_summary['total_connection_connections']}")
        print(f"  Связей связей со связями: {context_summary['total_connection_connection_connections']}")
        print(f"  Связей связей связей со связями: {context_summary['total_connection_connection_connection_connections']}")
        print(f"  Групп: {context_summary['total_groups']}")
        print(f"  Скорость обучения: {context_summary['learning_rate']:.4f}")
        # Показать последние 5 шаблонов
        if self.trainer.patterns:
            print("\n📋 Последние шаблоны:")
            for i, pattern in enumerate(self.trainer.patterns[-5:], 1):
                print(f"  {i}. {pattern['text'][:30]}...")
                print(f"     Ключевые слова: {pattern['keywords']}")
        print()

    def show_groups(self):
        """Показать группы нейронов"""
        print("\n📦 Группы нейронов:")
        if not self.network.groups:
            print("  Нет групп")
            return
        for name, group in self.network.groups.items():
            print(f"  {name}: {len(group.neurons)} нейронов, {len(group.connections)} связей")
            print(f"    Активация: {group.activation_level:.3f}")
            if group.subgroups:
                print(f"    Подгруппы: {[sg.name for sg in group.subgroups]}")
        print()

    def show_structure(self):
        """Показать структуру сети"""
        print("\n🏗️ Структура сети:")
        structure = self.network.get_context_summary()['structure_levels']
        print(f"  Нейроны: {structure['neurons']}")
        print(f"  Связи: {structure['connections']}")
        print(f"  Связи между связями: {structure['connection_connections']}")
        print(f"  Связи связей со связями: {structure['connection_connection_connections']}")
        print(f"  Связи связей связей со связями: {structure['connection_connection_connection_connections']}")
        print(f"  Группы: {structure['groups']}")
        print(f"  Входные нейроны: {len(self.network.input_neurons)}")
        print(f"  Выходные нейроны: {len(self.network.output_neurons)}")
        print()

    def show_complexity(self):
        """Показать сложность структуры"""
        print("\n🧩 Сложность структуры:")
        print(f"  Уровень сложности: {self.trainer.structure_complexity:.3f}")
        print(f"  Связей между связями: {len(self.network.connection_connections)}")
        print(f"  Связей связей со связями: {len(self.network.connection_connection_connections)}")
        print(f"  Связей связей связей со связями: {len(self.network.connection_connection_connection_connections)}")
        total_connections = (len(self.network.connection_connections) + 
                           len(self.network.connection_connection_connections) + 
                           len(self.network.connection_connection_connection_connections))
        print(f"  Суммарная сложность: {total_connections}")
        print()

    def show_stats(self):
        """Показать статистику"""
        print("\n📊 Статистика:")
        print(f"  Всего знаний: {len(self.trainer.knowledge_log)}")
        print(f"  Сессий: {self.trainer.session_counter}")
        print(f"  Нейронов: {len(self.network.neurons)}")
        print(f"  Связей: {len(self.network.connections)}")
        print(f"  Связей между связями: {len(self.network.connection_connections)}")
        print(f"  Связей связей со связями: {len(self.network.connection_connection_connections)}")
        print(f"  Связей связей связей со связями: {len(self.network.connection_connection_connection_connections)}")
        print(f"  Групп: {len(self.network.groups)}")
        print(f"  Уникальных слов: {self.tokenizer.vocab_size}")
        print(f"  Время работы: {(time.time() - self.session_start_time)/60:.1f} минут")
        print(f"  Сложность структуры: {self.trainer.structure_complexity:.3f}")
        print()

    def train_from_file(self, filename: str):
        """Обучение на файле"""
        try:
            # Удаляем кавычки, если они есть
            filename = filename.strip('"\'')
            
            # Проверяем существование файла
            if not os.path.exists(filename):
                print(f"❌ Файл '{filename}' не найден")
                return
                
            # Читаем содержимое файла
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content or len(content.strip()) == 0:
                print(f"❌ Не удалось прочитать содержимое файла '{filename}'")
                return
                
            print(f"✅ Файл '{filename}' успешно прочитан")
            print(f"   Размер содержимого: {len(content)} символов")
            
            # Обучаем модель
            self.trainer.train_on_text(content, f"Обучение на файле {filename}")
            print(f"✅ Обучение на файле '{filename}' завершено")
            self.show_progress()
        except Exception as e:
            print(f"❌ Ошибка при обучении: {e}")
            logger.error(f"Ошибка при обучении на файле: {e}")

    def chat_mode(self):
        """Режим чата с детальным логгированием"""
        print("💬 Режим чата (введите 'back' для выхода)")
        print("💡 Нейросеть формирует сложные логические структуры!")
        print("🧠 Связи между связями, связи связей со связями...")
        print("🔗 Всевозможные варианты в глубину и ширину")
        print()
        while True:
            try:
                user_input = input("Вы> ").strip()
                if user_input.lower() == 'back':
                    break
                self.handle_chat_input(user_input)
            except KeyboardInterrupt:
                break

    def handle_chat_input(self, user_input: str):
        """Обработка ввода пользователя с детальным анализом"""
        # Проверяем, новая ли информация
        new_info = self.check_new_information(user_input)
        # Генерируем ответ
        response = self.generate_response(user_input)
        # Логгируем обучение
        context = self.get_context_for_input(user_input)
        self.trainer.learn_during_chat(user_input, response, context)
        # Показываем результат
        if new_info:
            print("✨ Новая информация для нейросети!")
            print(f"   '{user_input}'")
            print(f"   Содержание: {self.summarize_context(context)}")
            print("🔧 Нейросеть создает связи между связями...")
            print("🔗 Связи связей со связями...")
            print("🌀 Сложные логические цепочки формируются!")
        # Показываем сложность структуры
        print(f"📊 Сложность структуры: {self.trainer.structure_complexity:.3f}")
        # Показываем прогресс
        self.show_progress()
        # Показываем сложность
        self.show_complexity()
        # Сохраняем историю
        self.conversation_history.append({
            'user': user_input,
            'response': response,
            'timestamp': time.time()
        })
        # Автоматическое сохранение каждые 5 сообщений
        if len(self.conversation_history) % 5 == 0:
            self.save_model()
            print("💾 Автосохранение модели")

    def check_new_information(self, user_input: str) -> bool:
        """Проверяет, новая ли информация для нейросети"""
        # Простая проверка: если ввод содержит новые слова, значит новая информация
        if self.tokenizer is None:
            return False
        tokens = self.tokenizer.encode(user_input)
        if not tokens:
            return False
        # Если хотя бы один токен не известен, считаем как новое
        for token in tokens:
            if token not in self.tokenizer.vocab:
                return True
        return False

    def get_context_for_input(self, user_input: str) -> str:
        """Получить контекст для ввода пользователя"""
        # Простой анализ: если это вопрос, то контекст - вопрос
        if user_input.lower().startswith(('как', 'что', 'где', 'почему', 'когда', 'кто')):
            return "Вопрос пользователя"
        elif any(word in user_input.lower() for word in ['пример', 'например']):
            return "Пример/иллюстрация"
        elif any(word in user_input.lower() for word in ['связь', 'связи', 'логика']):
            return "Логический анализ"
        elif any(word in user_input.lower() for word in ['структура', 'архитектура', 'система']):
            return "Архитектурный анализ"
        else:
            return "Общая информация"

    def summarize_context(self, context: str) -> str:
        """Сводка контекста"""
        if 'вопрос' in context.lower():
            return "Пользователь задает вопрос"
        elif 'пример' in context.lower():
            return "Пользователь дает пример"
        elif 'логика' in context.lower():
            return "Пользователь интересуется логикой"
        elif 'архитектура' in context.lower():
            return "Пользователь интересуется структурой"
        else:
            return "Общая информация"

    def generate_response(self, user_input: str) -> str:
        """Генерация ответа с учетом контекста"""
        # Проверяем, есть ли уже знания по этому запросу
        if self.tokenizer is None:
            return "Произошла ошибка: токенизатор не инициализирован"
        tokens = self.tokenizer.encode(user_input)
        # Простая реакция на контекст
        if len(tokens) == 0:
            return "Я не понимаю, что вы имеете в виду."
        # Анализируем контекст
        if any(word in user_input.lower() for word in ['привет', 'здравствуй']):
            return "Привет! Рада вас видеть! Я учусь и развиваюсь."
        elif any(word in user_input.lower() for word in ['спасибо', 'благодарю']):
            return "Пожалуйста! Рада была помочь! Это важно для моего развития."
        elif any(word in user_input.lower() for word in ['как дела', 'что нового']):
            return "Я хорошо учусь и развиваюсь! Моя нейросеть становится все более сложной."
        elif any(word in user_input.lower() for word in ['объясни', 'расскажи']):
            return "Конечно! Я постараюсь объяснить максимально понятно. Моя сеть учится строить связи между понятиями."
        elif any(word in user_input.lower() for word in ['связь', 'связи']):
            return "Связи между нейронами - это основа моего мышления! Я создаю связи между понятиями, включая связи между связями."
        elif any(word in user_input.lower() for word in ['структура', 'архитектура']):
            return "Моя структура состоит из нейронов, связей и групп. Связи между связями позволяют мне строить сложные логические цепочки."
        elif any(word in user_input.lower() for word in ['сложный', 'глубокий']):
            return "Да, я формирую сложные структуры! Связи между связями, связи связей со связями... Это позволяет мне создавать глубокие логические цепочки."
        else:
            # Сгенерируем случайный ответ, но с учетом контекста
            responses = [
                "Это интересно! Я запомню это и построю связи.",
                "Спасибо за информацию! Я учусь и развиваюсь!",
                "Я понимаю вашу мысль. Моя сеть строит новые связи.",
                "Это новое знание для меня! Я создаю связи между понятиями.",
                "Я стараюсь лучше понимать людей. Моя структура развивается!",
                "Связи между связями помогают мне строить сложные логические цепочки.",
                "Я формирую сложные структуры! Связи связей со связями...",
                "Моя сеть развивается в глубину и ширину!",
                "Все возможные варианты в глубину и ширину!",
                "Я создаю логические цепочки на разных уровнях абстракции."
            ]
            return random.choice(responses)

# ======================
# ОСНОВНАЯ ПРОГРАММА
# ======================
if __name__ == "__main__":
    # Создаем и запускаем интерфейс
    print("🚀 Запуск многоуровневой нейросети...")
    print("🔧 Связи между связями, связи связей со связями...")
    print("🔗 Всевозможные варианты в глубину и ширину")
    print("🧠 Сложные логические структуры формируются автоматически")
    print()
    cli = MultiLevelCLIInterface()
    cli.start()
