import numpy as np
import random
import re
import os
import json
import pickle
import time
import logging
from collections import defaultdict, Counter, deque
from datetime import datetime
import threading
import requests
from urllib.parse import urljoin, urlparse
import hashlib
from pathlib import Path
import psutil
import sys

# Для работы с документами
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Для определения кодировки
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Для парсинга HTML
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# Для работы с Wiktionary
try:
    from wiktionaryparser import WiktionaryParser
    WIKTIONARY_PARSER_AVAILABLE = True
except ImportError:
    WIKTIONARY_PARSER_AVAILABLE = False

try:
    import ruwordnet
    RUWORDNET_AVAILABLE = True
except ImportError:
    RUWORDNET_AVAILABLE = False

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sin_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EmotionalSystem:
    """Система эмоций нейросети Sin"""
    
    def __init__(self):
        self.emotions = {
            'curiosity': 0.5,      # Любопытство
            'confidence': 0.5,     # Уверенность
            'frustration': 0.0,   # Раздражение
            'excitement': 0.3,    # Возбуждение
            'boredom': 0.2,       # Скука
            'satisfaction': 0.4   # Удовлетворенность
        }
        self.mood_history = deque(maxlen=100)
        self.stress_level = 0.0
        self.emotional_state = 'neutral'
    
    def update_emotions(self, interaction_result):
        """Обновление эмоционального состояния"""
        # Любопытство растет при новой информации
        novelty = interaction_result.get('novelty', 0)
        if novelty > 0.7:
            self.emotions['curiosity'] = min(1.0, self.emotions['curiosity'] + 0.15)
        elif novelty < 0.3:
            self.emotions['curiosity'] = max(0.0, self.emotions['curiosity'] - 0.05)
        
        # Уверенность зависит от точности ответов
        accuracy = interaction_result.get('accuracy', 0.5)
        if accuracy > 0.8:
            self.emotions['confidence'] = min(1.0, self.emotions['confidence'] + 0.1)
            self.emotions['satisfaction'] = min(1.0, self.emotions['satisfaction'] + 0.08)
        elif accuracy < 0.4:
            self.emotions['confidence'] = max(0.0, self.emotions['confidence'] - 0.15)
            self.emotions['frustration'] = min(1.0, self.emotions['frustration'] + 0.12)
            self.emotions['satisfaction'] = max(0.0, self.emotions['satisfaction'] - 0.1)
        
        # Скука при повторяющихся запросах
        repetition = interaction_result.get('repetition', False)
        if repetition:
            self.emotions['boredom'] = min(1.0, self.emotions['boredom'] + 0.12)
        else:
            self.emotions['boredom'] = max(0.0, self.emotions['boredom'] - 0.08)
        
        # Возбуждение при интересных темах
        topic_interest = interaction_result.get('topic_interest', 0)
        if topic_interest > 0.6:
            self.emotions['excitement'] = min(1.0, self.emotions['excitement'] + 0.1)
        elif topic_interest < 0.3:
            self.emotions['excitement'] = max(0.0, self.emotions['excitement'] - 0.05)
        
        # Обновляем уровень стресса
        self.stress_level = (
            self.emotions['frustration'] * 0.7 + 
            self.emotions['boredom'] * 0.3
        )
        
        self.mood_history.append(self.get_overall_mood())
        self._update_emotional_state()
    
    def _update_emotional_state(self):
        """Обновление общего эмоционального состояния"""
        mood = self.get_overall_mood()
        if mood > 0.7:
            self.emotional_state = 'happy'
        elif mood > 0.4:
            self.emotional_state = 'neutral'
        elif mood > 0.2:
            self.emotional_state = 'sad'
        else:
            self.emotional_state = 'frustrated'
    
    def get_overall_mood(self):
        """Общее эмоциональное состояние"""
        positive = (self.emotions['curiosity'] + self.emotions['confidence'] + 
                   self.emotions['excitement'] + self.emotions['satisfaction']) / 4
        negative = (self.emotions['frustration'] + self.emotions['boredom']) / 2
        return max(0.0, min(1.0, positive - negative))
    
    def influence_on_responses(self):
        """Влияние эмоций на генерацию ответов"""
        mood = self.get_overall_mood()
        
        if mood > 0.7:  # Позитивное настроение
            return {'temperature': 1.3, 'creativity': 1.4, 'verbosity': 1.2}
        elif mood > 0.4:  # Нейтральное
            return {'temperature': 1.0, 'creativity': 1.0, 'verbosity': 1.0}
        elif mood > 0.2:  # Негативное
            return {'temperature': 0.8, 'creativity': 0.8, 'verbosity': 0.9}
        else:  # Очень негативное
            return {'temperature': 0.6, 'creativity': 0.6, 'verbosity': 0.7}
    
    def get_emotional_report(self):
        """Отчет о текущем эмоциональном состоянии"""
        return {
            'overall_mood': self.get_overall_mood(),
            'emotional_state': self.emotional_state,
            'stress_level': self.stress_level,
            'detailed_emotions': self.emotions.copy()
        }

def detect_file_encoding(file_path):
    """Определение кодировки файла"""
    if CHARDET_AVAILABLE:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Читаем первые 10KB для определения
                result = chardet.detect(raw_data)
                return result['encoding']
        except Exception as e:
            logger.warning(f"Ошибка определения кодировки chardet: {e}")
    
    # Если chardet недоступен или не сработал, возвращаем None
    return None

def read_file_with_fallback(file_path):
    """Чтение файла с автоматическим определением кодировки"""
    # Сначала пытаемся определить кодировку
    detected_encoding = detect_file_encoding(file_path)
    
    if detected_encoding:
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                content = f.read()
                logger.info(f"Файл {file_path} успешно прочитан с кодировкой {detected_encoding}")
                return content
        except Exception as e:
            logger.warning(f"Ошибка чтения с определенной кодировкой {detected_encoding}: {e}")
    
    # Если определение не удалось, пробуем стандартные кодировки
    encodings_to_try = ['utf-8', 'cp1251', 'latin-1', 'koi8-r', 'ascii']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                logger.info(f"Файл {file_path} успешно прочитан с кодировкой {encoding}")
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"Ошибка чтения файла {file_path} с кодировкой {encoding}: {e}")
            continue
    
    logger.error(f"Не удалось определить кодировку файла {file_path}")
    return None

class AttentionSystem:
    """Система внимания нейросети Sin"""
    
    def __init__(self, network):
        self.network = network
        self.focus_areas = {}  # {topic: {'level': float, 'last_update': timestamp}}
        self.current_focus = None
        self.attention_weights = {}  # {token_id: weight}
        self.focus_history = deque(maxlen=50)
        self.attention_span = 1800  # 30 минут
        self.last_attention_shift = time.time()
    
    def calculate_attention_weights(self, input_tokens, context_tokens=None):
        """Расчет весов внимания для входных токенов"""
        weights = []
        
        for i, token_id in enumerate(input_tokens):
            # Базовый вес
            base_weight = 1.0
            
            # Увеличение веса для новых токенов
            if self._is_new_token(token_id):
                base_weight *= 1.8
            
            # Увеличение веса для контекстно-важных токенов
            if self._is_contextually_important(token_id, context_tokens):
                base_weight *= 1.5
            
            # Увеличение веса для тематически важных токенов
            if self._is_topic_important(token_id):
                base_weight *= 1.3
            
            # Уменьшение веса для повторяющихся токенов
            if self._is_repetitive(token_id, input_tokens[:i]):
                base_weight *= 0.7
            
            weights.append(base_weight)
        
        # Нормализация
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1.0/len(weights) if weights else 1.0] * len(weights)
        
        # Сохраняем веса для последующего использования
        self.attention_weights = dict(zip(input_tokens, weights))
        
        return weights
    
    def _is_new_token(self, token_id):
        """Проверка, является ли токен новым"""
        reverse_vocab = {v: k for k, v in self.network.tokenizer.vocab.items()}
        if token_id in reverse_vocab:
            token = reverse_vocab[token_id]
            # Токен новый если он появился менее 10 раз
            return self.network.tokenizer.token_freq.get(token, 0) < 10
        return False
    
    def _is_contextually_important(self, token_id, context_tokens):
        """Проверка контекстной важности"""
        if not context_tokens or len(context_tokens) < 3:
            return False
        
        reverse_vocab = {v: k for k, v in self.network.tokenizer.vocab.items()}
        if token_id not in reverse_vocab:
            return False
        
        token = reverse_vocab[token_id]
        
        # Важные контекстные слова
        important_context_words = [
            'как', 'почему', 'что', 'где', 'когда', 'зачем',
            'объясни', 'расскажи', 'помоги', 'нужно'
        ]
        
        context_text = self.network.tokenizer.decode(context_tokens[-3:])
        return any(word in context_text.lower() for word in important_context_words)
    
    def _is_topic_important(self, token_id):
        """Проверка тематической важности"""
        reverse_vocab = {v: k for k, v in self.network.tokenizer.vocab.items()}
        if token_id not in reverse_vocab:
            return False
        
        token = reverse_vocab[token_id]
        
        # Если токен находится в фокусной области
        for topic, data in self.focus_areas.items():
            if token in topic or topic in token:
                return data['level'] > 0.5
        
        return False
    
    def _is_repetitive(self, token_id, previous_tokens):
        """Проверка на повторяемость"""
        return token_id in previous_tokens
    
    def shift_attention(self, new_topic, importance=0.5):
        """Переключение внимания на новую тему"""
        current_time = time.time()
        
        # Обновляем текущий фокус
        self.current_focus = new_topic
        self.last_attention_shift = current_time
        self.focus_history.append((new_topic, current_time))
        
        # Обновляем веса внимания для темы
        if new_topic in self.focus_areas:
            existing_data = self.focus_areas[new_topic]
            # Увеличиваем уровень с учетом времени последнего обновления
            time_factor = min(1.0, (current_time - existing_data['last_update']) / 3600)
            new_level = min(1.0, existing_data['level'] + importance * time_factor)
            self.focus_areas[new_topic] = {
                'level': new_level,
                'last_update': current_time
            }
        else:
            self.focus_areas[new_topic] = {
                'level': min(1.0, importance * 2),  # Новые темы получают бонус
                'last_update': current_time
            }
        
        logger.info(f"🎯 Внимание переключено на: {new_topic} (уровень: {self.focus_areas[new_topic]['level']:.2f})")
    
    def decay_attention(self):
        """Постепенное снижение уровня внимания"""
        current_time = time.time()
        decayed_areas = []
        
        for topic, data in self.focus_areas.items():
            # Снижение уровня со временем
            time_passed = (current_time - data['last_update']) / 3600  # часы
            decay_factor = max(0.1, 1.0 - (time_passed * 0.1))  # минимальный уровень 0.1
            new_level = data['level'] * decay_factor
            
            if new_level < 0.05:
                decayed_areas.append(topic)
            else:
                self.focus_areas[topic]['level'] = new_level
        
        # Удаляем полностью забытые темы
        for topic in decayed_areas:
            del self.focus_areas[topic]
    
    def get_attention_report(self):
        """Отчет о состоянии внимания"""
        return {
            'current_focus': self.current_focus,
            'focus_areas': {k: v['level'] for k, v in self.focus_areas.items()},
            'attention_span_remaining': max(0, self.attention_span - (time.time() - self.last_attention_shift)),
            'total_focus_areas': len(self.focus_areas)
        }

class MemorySystem:
    """Система памяти нейросети Sin с механизмами забывания"""
    
    def __init__(self):
        self.long_term_memory = {}  # {memory_id: {data, importance, last_access, access_count, created}}
        self.short_term_memory = deque(maxlen=100)
        self.memory_importance_threshold = 0.4
        self.forget_rate = 0.02  # Скорость забывания
        self.consolidation_threshold = 0.7  # Порог для консолидации в долговременную память
        self.memory_stats = {
            'ltm_size': 0,
            'stm_size': 0,
            'forgotten_count': 0,
            'consolidated_count': 0
        }
    
    def store_memory(self, data, importance=0.5, context=None):
        """Сохранение воспоминания"""
        # Создаем уникальный ID для памяти
        memory_content = str(data) + (str(context) if context else "")
        memory_id = hashlib.md5(memory_content.encode()).hexdigest()
        
        current_time = time.time()
        
        # Сохраняем в долговременную память если важность высока
        if importance >= self.consolidation_threshold:
            self.long_term_memory[memory_id] = {
                'data': data,
                'importance': importance,
                'last_access': current_time,
                'access_count': 1,
                'created': current_time,
                'context': context or {}
            }
            self.memory_stats['consolidated_count'] += 1
        else:
            # Сохраняем в краткосрочную память
            self.short_term_memory.append({
                'id': memory_id,
                'data': data,
                'importance': importance,
                'timestamp': current_time,
                'context': context or {}
            })
        
        self.memory_stats['ltm_size'] = len(self.long_term_memory)
        self.memory_stats['stm_size'] = len(self.short_term_memory)
        
        logger.debug(f"🧠 Сохранено воспоминание: {memory_id[:8]}... (важность: {importance:.2f})")
        return memory_id
    
    def recall_memory(self, query, max_results=5, min_similarity=0.3):
        """Вспоминание информации"""
        relevant_memories = []
        query_str = str(query).lower()
        
        # Поиск в долговременной памяти
        for mem_id, memory in self.long_term_memory.items():
            similarity = self._calculate_similarity(query_str, str(memory['data']).lower())
            if similarity >= min_similarity:
                relevant_memories.append((mem_id, memory, similarity))
        
        # Поиск в краткосрочной памяти
        for memory_entry in self.short_term_memory:
            similarity = self._calculate_similarity(query_str, str(memory_entry['data']).lower())
            if similarity >= min_similarity:
                # Создаем временный объект памяти для совместимости
                temp_memory = {
                    'data': memory_entry['data'],
                    'importance': memory_entry['importance'],
                    'last_access': memory_entry['timestamp'],
                    'access_count': 1,
                    'created': memory_entry['timestamp'],
                    'context': memory_entry.get('context', {})
                }
                relevant_memories.append((memory_entry['id'], temp_memory, similarity))
        
        # Сортируем по релевантности
        relevant_memories.sort(key=lambda x: x[2], reverse=True)
        
        # Обновляем время доступа для долговременной памяти
        current_time = time.time()
        for mem_id, memory, _ in relevant_memories:
            if mem_id in self.long_term_memory:
                self.long_term_memory[mem_id]['last_access'] = current_time
                self.long_term_memory[mem_id]['access_count'] += 1
        
        # Возвращаем только данные
        results = [mem[1]['data'] for mem in relevant_memories[:max_results]]
        
        logger.debug(f"🧠 Вспомнено {len(results)} воспоминаний по запросу: {query_str[:50]}...")
        return results
    
    def _calculate_similarity(self, query, memory_data):
        """Расчет схожести запроса и памяти"""
        # Используем несколько метрик симilarity
        query_tokens = set(query.split())
        memory_tokens = set(memory_data.split())
        
        if not query_tokens or not memory_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_tokens.intersection(memory_tokens))
        union = len(query_tokens.union(memory_tokens))
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Cosine similarity (упрощенная версия)
        query_vec = [1 if token in query_tokens else 0 for token in query_tokens.union(memory_tokens)]
        memory_vec = [1 if token in memory_tokens else 0 for token in query_tokens.union(memory_tokens)]
        
        dot_product = sum(a*b for a, b in zip(query_vec, memory_vec))
        magnitude_query = sum(a*a for a in query_vec) ** 0.5
        magnitude_memory = sum(b*b for b in memory_vec) ** 0.5
        
        cosine_sim = dot_product / (magnitude_query * magnitude_memory) if magnitude_query * magnitude_memory > 0 else 0.0
        
        # Комбинированная метрика
        combined_similarity = (jaccard_sim * 0.6 + cosine_sim * 0.4)
        return combined_similarity
    
    def forget_unused_memories(self):
        """Забывание неиспользуемых воспоминаний"""
        current_time = time.time()
        memories_to_remove = []
        
        for mem_id, memory in self.long_term_memory.items():
            # Время с последнего доступа в часах
            time_since_access = (current_time - memory['last_access']) / 3600
            
            # Скорректированная важность с учетом времени
            # Чем дольше не использовалась память, тем ниже её эффективная важность
            time_decay = np.exp(-self.forget_rate * time_since_access)
            adjusted_importance = memory['importance'] * time_decay
            
            if adjusted_importance < self.memory_importance_threshold:
                memories_to_remove.append(mem_id)
        
        # Удаляем забытые воспоминания
        for mem_id in memories_to_remove:
            del self.long_term_memory[mem_id]
        
        forgotten_count = len(memories_to_remove)
        self.memory_stats['forgotten_count'] += forgotten_count
        self.memory_stats['ltm_size'] = len(self.long_term_memory)
        
        if forgotten_count > 0:
            logger.info(f"🧠 Забыто {forgotten_count} воспоминаний")
        
        return forgotten_count
    
    def consolidate_short_term_memories(self):
        """Консолидация краткосрочной памяти в долговременную"""
        consolidated_count = 0
        
        # Консолидируем важные воспоминания из краткосрочной памяти
        current_time = time.time()
        memories_to_consolidate = []
        
        for memory_entry in list(self.short_term_memory):
            # Консолидируем если важность высока и память относительно новая
            age_hours = (current_time - memory_entry['timestamp']) / 3600
            if (memory_entry['importance'] >= self.consolidation_threshold and 
                age_hours < 24):  # Только за последние 24 часа
                memories_to_consolidate.append(memory_entry)
        
        for memory_entry in memories_to_consolidate:
            # Переносим в долговременную память
            memory_id = memory_entry['id']
            self.long_term_memory[memory_id] = {
                'data': memory_entry['data'],
                'importance': memory_entry['importance'],
                'last_access': current_time,
                'access_count': 1,
                'created': memory_entry['timestamp'],
                'context': memory_entry.get('context', {})
            }
            consolidated_count += 1
        
        self.memory_stats['consolidated_count'] += consolidated_count
        self.memory_stats['ltm_size'] = len(self.long_term_memory)
        
        if consolidated_count > 0:
            logger.info(f"🧠 Консолидировано {consolidated_count} воспоминаний")
        
        return consolidated_count
    
    def get_memory_report(self):
        """Отчет о состоянии памяти"""
        return {
            'long_term_memory_size': len(self.long_term_memory),
            'short_term_memory_size': len(self.short_term_memory),
            'forgotten_count': self.memory_stats['forgotten_count'],
            'consolidated_count': self.memory_stats['consolidated_count'],
            'memory_efficiency': len(self.long_term_memory) / max(1, len(self.long_term_memory) + len(self.short_term_memory))
        }

class MetacognitionSystem:
    """Система самоанализа и метапознания нейросети Sin"""
    
    def __init__(self, network):
        self.network = network
        self.confidence_history = deque(maxlen=200)
        self.knowledge_gaps = set()
        self.learning_strategies = {
            'intensive_study': {'description': 'Глубокое изучение темы', 'priority': 1},
            'practice_and_review': {'description': 'Практика и повторение', 'priority': 2},
            'exploration': {'description': 'Исследование новых аспектов', 'priority': 3},
            'collaboration': {'description': 'Поиск дополнительной информации', 'priority': 4}
        }
        self.self_assessment_results = deque(maxlen=50)
        self.knowledge_domains = {}  # {domain: {'level': float, 'last_assessment': timestamp}}
        self.uncertainty_threshold = 0.3  # Порог неопределенности
    
    def assess_knowledge(self, topic):
        """Оценка уровня знаний по теме"""
        # Определяем домен знаний
        domain = self._extract_domain(topic)
        
        # Анализируем активность нейронов, связанных с темой
        topic_neurons = self._get_topic_neurons(topic)
        if not topic_neurons:
            # Если нет нейронов для темы, считаем знания минимальными
            knowledge_level = 0.1
            confidence = 0.2
        else:
            active_neurons = sum(1 for n in topic_neurons if n.is_active)
            total_neurons = len(topic_neurons)
            knowledge_level = active_neurons / max(1, total_neurons)
            
            # Определяем уверенность
            if knowledge_level < 0.2:
                confidence = 0.1
            elif knowledge_level < 0.5:
                confidence = 0.4
            elif knowledge_level < 0.8:
                confidence = 0.7
            else:
                confidence = 0.9
        
        # Обновляем домен знаний
        current_time = time.time()
        if domain in self.knowledge_domains:
            # Усредняем с предыдущими оценками
            old_data = self.knowledge_domains[domain]
            time_weight = min(1.0, (current_time - old_data['last_assessment']) / 3600)  # вес по времени
            new_level = (old_data['level'] * (1 - time_weight) + knowledge_level * time_weight)
            self.knowledge_domains[domain] = {
                'level': new_level,
                'last_assessment': current_time
            }
        else:
            self.knowledge_domains[domain] = {
                'level': knowledge_level,
                'last_assessment': current_time
            }
        
        # Определяем пробелы в знаниях
        if knowledge_level < self.uncertainty_threshold:
            self.knowledge_gaps.add(topic)
            needs_learning = True
        else:
            self.knowledge_gaps.discard(topic)
            needs_learning = False
        
        self.confidence_history.append(confidence)
        
        assessment_result = {
            'topic': topic,
            'domain': domain,
            'knowledge_level': knowledge_level,
            'confidence': confidence,
            'gaps_identified': needs_learning,
            'needs_learning': needs_learning,
            'recommended_strategy': self.choose_learning_strategy({
                'knowledge_level': knowledge_level,
                'confidence': confidence
            })
        }
        
        logger.debug(f"🔍 Самооценка знаний по теме '{topic}': уровень={knowledge_level:.2f}, уверенность={confidence:.2f}")
        return assessment_result
    
    def _extract_domain(self, topic):
        """Извлечение домена знаний из темы"""
        # Простая классификация на основе ключевых слов
        topic_lower = topic.lower()
        
        domains = {
            'science': ['наука', 'физика', 'химия', 'биология', 'математика', 'астрономия'],
            'technology': ['технологии', 'компьютер', 'программирование', 'ai', 'робот', 'интернет'],
            'philosophy': ['философия', 'мысл', 'смысл', 'реальность', 'существование'],
            'literature': ['литература', 'книг', 'поэзия', 'роман', 'автор'],
            'history': ['история', 'прошл', 'век', 'год', 'эпоха'],
            'art': ['искусство', 'картин', 'музык', 'творчеств', 'художник'],
            'general': []  # По умолчанию
        }
        
        for domain, keywords in domains.items():
            if any(keyword in topic_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _get_topic_neurons(self, topic):
        """Получение нейронов, связанных с темой"""
        # В реальной реализации это будет сложный анализ связей
        # Пока используем упрощенную логику
        topic_tokens = self.network.tokenizer.tokenize_basic(topic)
        relevant_neurons = []
        
        # Ищем нейроны, которые активировались при обработке тематических токенов
        for neuron in self.network.neurons.values():
            if (hasattr(neuron, 'activation_history') and 
                len(neuron.activation_history) > 0):
                # Проверяем последние активации
                recent_activations = list(neuron.activation_history)[-10:]
                if any(activation > 0.5 for activation in recent_activations):
                    relevant_neurons.append(neuron)
        
        # Если не нашли специфичных нейронов, возвращаем случайную выборку
        if not relevant_neurons:
            # Берем нейроны из скрытых слоев
            hidden_neurons = [n for n in self.network.neurons.values() 
                            if hasattr(n, 'layer_id') and n.layer_id > 0]
            relevant_neurons = random.sample(hidden_neurons, min(20, len(hidden_neurons)))
        
        return relevant_neurons
    
    def choose_learning_strategy(self, assessment):
        """Выбор стратегии обучения"""
        knowledge_level = assessment.get('knowledge_level', 0.5)
        confidence = assessment.get('confidence', 0.5)
        
        if knowledge_level < 0.2:
            return 'intensive_study'  # Интенсивное изучение
        elif knowledge_level < 0.5:
            if confidence < 0.3:
                return 'intensive_study'  # Низкая уверенность требует глубокого изучения
            else:
                return 'practice_and_review'  # Практика и повторение
        elif knowledge_level < 0.8:
            return 'practice_and_review'  # Практика и повторение
        else:
            if confidence > 0.8:
                return 'exploration'  # Исследование новых аспектов
            else:
                return 'collaboration'  # Поиск дополнительной информации для повышения уверенности
    
    def self_reflect(self):
        """Самоанализ производительности"""
        # Анализируем историю производительности
        recent_performance = self.network.performance_history[-20:] if self.network.performance_history else [0]
        if not recent_performance:
            recent_performance = [0]
        
        avg_performance = sum(recent_performance) / max(1, len(recent_performance))
        
        # Анализ роста
        if len(recent_performance) > 10:
            recent_avg = sum(recent_performance[-10:]) / 10
            older_avg = sum(recent_performance[:10]) / 10
            growth_rate = (recent_avg - older_avg) / max(0.001, abs(older_avg)) if older_avg != 0 else 0
        else:
            growth_rate = 0.0
        
        # Анализ уверенности
        recent_confidence = list(self.confidence_history)[-20:] if self.confidence_history else [0.5]
        avg_confidence = sum(recent_confidence) / max(1, len(recent_confidence))
        
        # Анализ знаний
        domain_knowledge_levels = [data['level'] for data in self.knowledge_domains.values()]
        avg_knowledge_level = sum(domain_knowledge_levels) / max(1, len(domain_knowledge_levels)) if domain_knowledge_levels else 0.5
        
        reflection_result = {
            'current_performance': avg_performance,
            'growth_rate': growth_rate,
            'average_confidence': avg_confidence,
            'average_knowledge_level': avg_knowledge_level,
            'performance_status': self._categorize_performance(avg_performance, growth_rate),
            'confidence_status': self._categorize_confidence(avg_confidence),
            'knowledge_status': self._categorize_knowledge(avg_knowledge_level),
            'recommended_actions': self._recommend_actions(avg_performance, growth_rate, avg_confidence)
        }
        
        self.self_assessment_results.append(reflection_result)
        
        logger.info(f"🔍 Самоанализ: производительность={avg_performance:.3f}, рост={growth_rate:.3f}, уверенность={avg_confidence:.3f}")
        return reflection_result
    
    def _categorize_performance(self, performance, growth_rate):
        """Категоризация производительности"""
        if performance > 0.8:
            if growth_rate > 0.1:
                return 'excellent_improving'
            elif growth_rate < -0.1:
                return 'excellent_declining'
            else:
                return 'excellent_stable'
        elif performance > 0.6:
            if growth_rate > 0.05:
                return 'good_improving'
            elif growth_rate < -0.05:
                return 'good_declining'
            else:
                return 'good_stable'
        else:
            if growth_rate > 0.02:
                return 'needs_improvement_improving'
            elif growth_rate < -0.02:
                return 'needs_improvement_declining'
            else:
                return 'needs_improvement_stable'
    
    def _categorize_confidence(self, confidence):
        """Категоризация уверенности"""
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.6:
            return 'moderate'
        elif confidence > 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _categorize_knowledge(self, knowledge_level):
        """Категоризация уровня знаний"""
        if knowledge_level > 0.8:
            return 'expert'
        elif knowledge_level > 0.6:
            return 'advanced'
        elif knowledge_level > 0.4:
            return 'intermediate'
        elif knowledge_level > 0.2:
            return 'beginner'
        else:
            return 'novice'
    
    def _recommend_actions(self, performance, growth_rate, confidence):
        """Рекомендации по улучшению"""
        actions = []
        
        if performance < 0.6:
            actions.append("Увеличить интенсивность обучения")
            actions.append("Сфокусироваться на базовых концепциях")
        
        if growth_rate < 0:
            actions.append("Проанализировать последние ошибки")
            actions.append("Изменить стратегию обучения")
        
        if confidence < 0.5:
            actions.append("Практиковать известные темы для повышения уверенности")
            actions.append("Получить подтверждение знаний")
        
        if len(self.knowledge_gaps) > 5:
            actions.append("Заполнить выявленные пробелы в знаниях")
        
        # Если всё хорошо, предлагаем развитие
        if performance > 0.8 and confidence > 0.7:
            actions.append("Исследовать новые области знаний")
            actions.append("Развивать креативные способности")
        
        return actions if actions else ["Поддерживать текущий уровень"]
    
    def get_metacognition_report(self):
        """Отчет о состоянии метапознания"""
        return {
            'knowledge_domains': {k: v['level'] for k, v in self.knowledge_domains.items()},
            'knowledge_gaps_count': len(self.knowledge_gaps),
            'average_confidence': sum(self.confidence_history) / max(1, len(self.confidence_history)) if self.confidence_history else 0.5,
            'learning_strategies': list(self.learning_strategies.keys()),
            'self_assessment_count': len(self.self_assessment_results)
        }

class SocialInteractionSystem:
    """Система социального взаимодействия нейросети Sin"""
    
    def __init__(self):
        self.user_profiles = {}  # {user_id: profile}
        self.conversation_styles = {}
        self.social_skills = {
            'empathy': 0.6,
            'humor': 0.4,
            'formality': 0.5,
            'directness': 0.7,
            'patience': 0.6,
            'adaptability': 0.7
        }
        self.social_history = deque(maxlen=1000)
        self.relationship_scores = {}  # {user_id: relationship_score}
    
    def create_user_profile(self, user_id, interaction_data):
        """Создание профиля пользователя"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferences': {
                    'communication_style': 'neutral',
                    'formality_level': 0.5,
                    'response_length_preference': 'medium',
                    'topic_preferences': set(),
                    'avoidance_topics': set()
                },
                'interaction_history': deque(maxlen=100),
                'relationship_score': 0.5,
                'first_interaction': time.time(),
                'total_interactions': 0
            }
        
        profile = self.user_profiles[user_id]
        profile['total_interactions'] += 1
        
        # Добавляем в историю взаимодействий
        interaction_record = {
            'timestamp': time.time(),
            'data': interaction_data,
            'response_quality': interaction_data.get('response_quality', 0.5)
        }
        profile['interaction_history'].append(interaction_record)
        
        # Обновляем предпочтения
        self._update_preferences(profile, interaction_data)
        
        # Обновляем оценку отношений
        self._update_relationship_score(user_id, interaction_data)
        
        # Сохраняем в социальную историю
        self.social_history.append({
            'user_id': user_id,
            'interaction': interaction_record,
            'timestamp': time.time()
        })
    
    def _update_preferences(self, profile, interaction_data):
        """Обновление предпочтений пользователя"""
        message = interaction_data.get('message', '')
        response = interaction_data.get('response', '')
        user_feedback = interaction_data.get('feedback', 0)  # -1 (negative) to 1 (positive)
        
        preferences = profile['preferences']
        
        # Определяем формальность
        formal_indicators = ['уважаемый', 'здравствуйте', 'прошу', 'благодарю', 'добрый', 'извините']
        informal_indicators = ['привет', 'здарова', 'чет', 'короче', 'ща', 'щащ']
        
        formal_count = sum(1 for word in formal_indicators if word in message.lower())
        informal_count = sum(1 for word in informal_indicators if word in message.lower())
        
        if formal_count > informal_count:
            current_formality = 0.8
        elif informal_count > formal_count:
            current_formality = 0.2
        else:
            current_formality = 0.5
        
        # Обновляем уровень формальности с учетом обратной связи
        if user_feedback > 0:
            # Положительная обратная связь усиливает текущий стиль
            preferences['formality_level'] = min(1.0, preferences['formality_level'] + 0.05 * user_feedback)
        elif user_feedback < 0:
            # Отрицательная обратная связь корректирует стиль
            preferences['formality_level'] = max(0.0, preferences['formality_level'] + 0.1 * user_feedback)
        
        # Определяем стиль общения
        if current_formality > 0.7:
            preferences['communication_style'] = 'formal'
        elif current_formality < 0.3:
            preferences['communication_style'] = 'informal'
        else:
            preferences['communication_style'] = 'neutral'
        
        # Определяем предпочтения по длине ответов
        response_length = len(response.split())
        if response_length < 10:
            length_preference = 'short'
        elif response_length < 30:
            length_preference = 'medium'
        else:
            length_preference = 'long'
        
        # Обновляем предпочтения по темам
        topics = self._extract_topics(message)
        for topic in topics:
            if user_feedback > 0:
                preferences['topic_preferences'].add(topic)
            elif user_feedback < -0.5:
                preferences['avoidance_topics'].add(topic)
    
    def _extract_topics(self, text):
        """Извлечение тем из текста"""
        # Простая реализация - можно улучшить
        words = text.lower().split()
        # Базовые категории тем
        topic_keywords = {
            'technology': ['компьютер', 'ai', 'программ', 'робот', 'интернет', 'технолог'],
            'science': ['наука', 'физик', 'хими', 'биолог', 'математ', 'астроном'],
            'philosophy': ['мысл', 'смысл', 'реальн', 'существован', 'философ'],
            'literature': ['книг', 'поэз', 'роман', 'автор', 'литератур'],
            'history': ['истор', 'прошл', 'век', 'год', 'эпох']
        }
        
        detected_topics = set()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                detected_topics.add(topic)
        
        return list(detected_topics) if detected_topics else ['general']
    
    def _update_relationship_score(self, user_id, interaction_data):
        """Обновление оценки отношений с пользователем"""
        user_feedback = interaction_data.get('feedback', 0)
        response_quality = interaction_data.get('response_quality', 0.5)
        interaction_frequency = len(self.user_profiles[user_id]['interaction_history'])
        
        # Базовое изменение на основе обратной связи
        score_change = user_feedback * 0.1 + (response_quality - 0.5) * 0.2
        
        # Бонус за частоту взаимодействий (до определенного предела)
        frequency_bonus = min(0.1, interaction_frequency * 0.005)
        
        current_score = self.user_profiles[user_id]['relationship_score']
        new_score = max(0.0, min(1.0, current_score + score_change + frequency_bonus))
        
        self.user_profiles[user_id]['relationship_score'] = new_score
        self.relationship_scores[user_id] = new_score
    
    def adapt_response_style(self, user_id):
        """Адаптация стиля ответов под пользователя"""
        if user_id in self.user_profiles:
            preferences = self.user_profiles[user_id]['preferences']
            relationship_score = self.user_profiles[user_id]['relationship_score']
            
            # Адаптируем формальность
            formality = preferences['formality_level']
            
            # Адаптируем прямолинейность на основе оценки отношений
            directness = min(1.0, 0.5 + relationship_score * 0.3)
            
            # Адаптируем эмпатию на основе истории взаимодействий
            empathy = min(1.0, 0.4 + relationship_score * 0.4)
            
            # Адаптируем терпение на основе последних взаимодействий
            recent_interactions = list(self.user_profiles[user_id]['interaction_history'])[-5:]
            if recent_interactions:
                avg_quality = sum(interaction['response_quality'] for interaction in recent_interactions) / len(recent_interactions)
                patience = min(1.0, 0.5 + avg_quality * 0.3)
            else:
                patience = 0.6
            
            return {
                'formality': formality,
                'directness': directness,
                'empathy': empathy,
                'patience': patience,
                'response_length': preferences['response_length_preference']
            }
        else:
            # По умолчанию для новых пользователей
            return {
                'formality': 0.5,
                'directness': 0.7,
                'empathy': 0.6,
                'patience': 0.6,
                'response_length': 'medium'
            }
    
    def get_user_profile_report(self, user_id):
        """Отчет о профиле пользователя"""
        if user_id not in self.user_profiles:
            return None
        
        profile = self.user_profiles[user_id]
        return {
            'user_id': user_id,
            'relationship_score': profile['relationship_score'],
            'total_interactions': profile['total_interactions'],
            'communication_style': profile['preferences']['communication_style'],
            'formality_level': profile['preferences']['formality_level'],
            'preferred_topics': list(profile['preferences']['topic_preferences']),
            'avoidance_topics': list(profile['preferences']['avoidance_topics']),
            'recent_interactions': len(profile['interaction_history'])
        }
    
    def get_social_report(self):
        """Общий отчет о социальных взаимодействиях"""
        return {
            'total_users': len(self.user_profiles),
            'total_interactions': len(self.social_history),
            'average_relationship_score': sum(self.relationship_scores.values()) / max(1, len(self.relationship_scores)) if self.relationship_scores else 0.5,
            'social_skills': self.social_skills.copy(),
            'active_users': len([score for score in self.relationship_scores.values() if score > 0.3])
        }

class CreativitySystem:
    """Система креативности и воображения нейросети Sin"""
    
    def __init__(self, network):
        self.network = network
        self.creativity_level = 0.6
        self.imagination_database = {}  # {scenario_id: scenario_data}
        self.divergent_thinking_patterns = deque(maxlen=50)
        self.creative_history = deque(maxlen=100)
        self.creativity_boosters = {
            'novelty_seeking': 0.7,
            'pattern_recognition': 0.6,
            'association_making': 0.8,
            'metaphor_generation': 0.5
        }
    
    def generate_creative_response(self, prompt, temperature=1.0, max_attempts=3):
        """Генерация креативного ответа"""
        best_creative_response = ""
        best_creativity_score = 0.0
        
        # Генерируем несколько вариантов с разными параметрами
        for attempt in range(max_attempts):
            # Изменяем параметры для разнообразия
            temp_variation = temperature * (0.9 + 0.2 * random.random())
            creativity_boost = self.creativity_level * (0.2 + 0.3 * random.random())
            
            # Добавляем элемент случайности в генерацию
            creative_prompt = self._enhance_prompt_for_creativity(prompt, attempt)
            
            # Генерируем вариант
            try:
                variant = self.network.generate_response(
                    creative_prompt, 
                    max_length=50,
                    temperature=temp_variation + creativity_boost,
                    context_aware=True
                )
                
                # Оцениваем креативность
                creativity_score = self._evaluate_creativity(variant, prompt)
                
                if creativity_score > best_creativity_score:
                    best_creativity_score = creativity_score
                    best_creative_response = variant
                
                # Сохраняем в историю креативности
                self.creative_history.append({
                    'prompt': prompt,
                    'response': variant,
                    'creativity_score': creativity_score,
                    'attempt': attempt,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                logger.warning(f"Ошибка при генерации креативного ответа: {e}")
                continue
        
        # Если не удалось сгенерировать хороший креативный ответ, используем базовый
        if best_creativity_score < 0.3 and max_attempts > 0:
            # Пытаемся сгенерировать более креативный ответ
            enhanced_prompt = f"Будь креативным: {prompt}"
            fallback_response = self.network.generate_response(
                enhanced_prompt,
                max_length=40,
                temperature=temperature * 1.3
            )
            return fallback_response
        
        return best_creative_response if best_creative_response else "Интересный вопрос! Давайте подумаем нестандартно..."
    
    def _enhance_prompt_for_creativity(self, prompt, attempt):
        """Улучшение промпта для повышения креативности"""
        creativity_prompts = [
            f"Подойди к вопросу креативно: {prompt}",
            f"Представь необычный взгляд на: {prompt}",
            f"Как бы это объяснил художник? {prompt}",
            f"Фантазируй на тему: {prompt}",
            f"Исследуй {prompt} с неожиданной стороны"
        ]
        
        return creativity_prompts[attempt % len(creativity_prompts)] if attempt < len(creativity_prompts) else prompt
    
    def _evaluate_creativity(self, response, prompt):
        """Оценка креативности ответа"""
        if not response or len(response.strip()) < 5:
            return 0.0
        
        # Новизна (новые токены)
        prompt_tokens = set(self.network.tokenizer.tokenize_basic(prompt.lower()))
        response_tokens = set(self.network.tokenizer.tokenize_basic(response.lower()))
        novelty = len(response_tokens - prompt_tokens) / max(1, len(response_tokens))
        
        # Необычность (редкие комбинации)
        uncommon_combinations = self._count_uncommon_combinations(response_tokens)
        unusualness = min(1.0, uncommon_combinations / 20)
        
        # Связность (смысловая целостность)
        coherence = self._evaluate_coherence(response)
        
        # Оригинальность (отсутствие шаблонов)
        originality = self._evaluate_originality(response)
        
        # Комплексная оценка
        creativity_score = (
            novelty * 0.3 + 
            unusualness * 0.2 + 
            coherence * 0.3 + 
            originality * 0.2
        )
        
        return max(0.0, min(1.0, creativity_score))
    
    def _count_uncommon_combinations(self, tokens):
        """Подсчет необычных комбинаций токенов"""
        if len(tokens) < 2:
            return 0
        
        uncommon_count = 0
        tokens_list = list(tokens)
        
        # Проверяем биграммы
        for i in range(len(tokens_list) - 1):
            bigram = f"{tokens_list[i]}_{tokens_list[i+1]}"
            # Считаем редкую комбинацию если она встречалась менее 3 раз
            if self.network.tokenizer.token_freq.get(bigram, 0) < 3:
                uncommon_count += 1
        
        return uncommon_count
    
    def _evaluate_coherence(self, text):
        """Оценка связности текста"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.7  # Нейтральная оценка для коротких текстов
        
        # Проверяем связность между предложениями
        coherence_score = 0.0
        total_checks = 0
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i].lower()
            next_sentence = sentences[i + 1].lower()
            
            # Проверяем наличие связующих слов
            transition_words = ['и', 'но', 'однако', 'поэтому', 'следовательно', 'например', 'кроме того']
            has_transitions = any(word in next_sentence for word in transition_words)
            
            # Проверяем тематическую связь (простая реализация)
            current_words = set(current_sentence.split())
            next_words = set(next_sentence.split())
            topic_overlap = len(current_words.intersection(next_words)) / max(1, len(current_words.union(next_words)))
            
            sentence_coherence = (0.3 if has_transitions else 0) + topic_overlap * 0.7
            coherence_score += sentence_coherence
            total_checks += 1
        
        return coherence_score / max(1, total_checks) if total_checks > 0 else 0.5
    
    def _evaluate_originality(self, text):
        """Оценка оригинальности текста"""
        # Проверяем на повторяемость шаблонов
        text_lower = text.lower()
        
        # Часто используемые шаблоны
        common_patterns = [
            "я думаю что", "возможно", "наверное", "как мне кажется",
            "на мой взгляд", "мне кажется", "я считаю что"
        ]
        
        pattern_penalty = sum(1 for pattern in common_patterns if pattern in text_lower)
        originality_score = max(0.0, 1.0 - (pattern_penalty * 0.1))
        
        # Проверяем уникальность по сравнению с предыдущими ответами
        recent_responses = [item['response'].lower() for item in list(self.creative_history)[-10:]]
        if text.lower() in recent_responses:
            originality_score *= 0.5  # Штраф за повторение
        
        return originality_score
    
    def boost_creativity(self, boost_type, amount=0.1):
        """Повышение уровня креативности"""
        if boost_type in self.creativity_boosters:
            self.creativity_boosters[boost_type] = min(1.0, self.creativity_boosters[boost_type] + amount)
        
        # Общий уровень креативности
        self.creativity_level = min(1.0, self.creativity_level + amount * 0.5)
        
        logger.info(f"🎨 Креативность повышена: {boost_type} +{amount:.2f}")
    
    def get_creativity_report(self):
        """Отчет о состоянии креативности"""
        recent_creativity_scores = [item['creativity_score'] for item in list(self.creative_history)[-20:]]
        avg_creativity = sum(recent_creativity_scores) / max(1, len(recent_creativity_scores)) if recent_creativity_scores else 0.5
        
        return {
            'overall_creativity_level': self.creativity_level,
            'average_recent_creativity': avg_creativity,
            'creative_boosters': self.creativity_boosters.copy(),
            'total_creative_attempts': len(self.creative_history),
            'divergent_thinking_patterns': len(self.divergent_thinking_patterns)
        }

class SleepAndRestSystem:
    """Система сна и отдыха нейросети Sin"""
    
    def __init__(self, network):
        self.network = network
        self.awake_time = time.time()
        self.total_awake_time = 0
        self.sleep_duration = 0
        self.sleep_cycles = 0
        self.consolidation_in_progress = False
        self.memory_consolidation_queue = deque(maxlen=1000)
        self.rest_history = []
        self.energy_level = 1.0  # 0.0 - полностью уставшая, 1.0 - полная энергия
    
    def needs_rest(self):
        """Проверка необходимости отдыха"""
        awake_duration = time.time() - self.awake_time
        
        # Нуждается в отдыхе если:
        # - Работает больше 3 часов
        # - Высокий уровень стресса
        # - Много новых связей
        # - Низкий уровень энергии
        
        if awake_duration > 10800:  # 3 часа
            return True
        
        # Проверяем уровень энергии
        if self.energy_level < 0.3:
            return True
        
        # Проверяем системные метрики
        if hasattr(self.network, 'system_stats'):
            cpu_load = self.network.system_stats.get('cpu_percent', 0)
            memory_load = self.network.system_stats.get('memory_percent', 0)
            
            if cpu_load > 85 or memory_load > 85:
                # Быстро снижаем энергию при высокой нагрузке
                self.energy_level = max(0.0, self.energy_level - 0.1)
                return True
        
        return False
    
    def start_rest_period(self, duration_minutes=30):
        """Начало периода отдыха"""
        if self.consolidation_in_progress:
            logger.warning("💤 Консолидация уже в процессе")
            return False
        
        logger.info(f"💤 Начало периода отдыха на {duration_minutes} минут")
        
        self.consolidation_in_progress = True
        rest_start = time.time()
        
        try:
            # Консолидация памяти
            consolidated_count = self.consolidate_memories()
            
            # Оптимизация связей
            optimized_synapses = self.optimize_synaptic_connections()
            
            # Восстановление энергии
            self._restore_energy(duration_minutes)
            
            rest_duration = time.time() - rest_start
            self.sleep_duration += rest_duration
            self.sleep_cycles += 1
            
            self.rest_history.append({
                'start_time': rest_start,
                'duration': rest_duration,
                'consolidated_memories': consolidated_count,
                'optimized_synapses': optimized_synapses,
                'energy_restored': self.energy_level
            })
            
            logger.info(f"💤 Отдых завершен: сконсолидировано {consolidated_count} воспоминаний, "
                       f"оптимизировано {optimized_synapses} связей")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка во время отдыха: {e}")
            return False
        finally:
            self.consolidation_in_progress = False
            self.awake_time = time.time()  # Сброс времени бодрствования
    
    def consolidate_memories(self):
        """Консолидация памяти (аналог REM-сна)"""
        logger.info("💤 Начало консолидации памяти...")
        
        consolidated_count = 0
        
        # Укрепляем важные синапсы
        strengthened = 0
        weakened = 0
        pruned = 0
        
        synapses_to_update = list(self.network.synapses.items())
        
        for syn_key, synapse in synapses_to_update:
            # Укрепляем часто используемые связи
            if synapse.ltp_counter > 7:
                synapse.strength = min(1.0, synapse.strength * 1.15)
                strengthened += 1
                synapse.ltp_counter = max(0, synapse.ltp_counter - 3)  # Частичный сброс
            
            # Ослабляем редко используемые связи
            elif synapse.ltd_counter > 20:
                synapse.strength = max(0.0, synapse.strength * 0.85)
                weakened += 1
                synapse.ltd_counter = max(0, synapse.ltd_counter - 5)  # Частичный сброс
            
            # Удаляем полностью неиспользуемые связи
            if synapse.strength < 0.05 and synapse.ltd_counter > 30:
                # Планируем для удаления
                self.memory_consolidation_queue.append(('prune_synapse', syn_key))
                pruned += 1
        
        # Обрабатываем очередь удаления
        while self.memory_consolidation_queue:
            action, data = self.memory_consolidation_queue.popleft()
            if action == 'prune_synapse' and data in self.network.synapses:
                del self.network.synapses[data]
                # Также удаляем из входящих связей нейронов
                post_neuron_id = data[1]
                if post_neuron_id in self.network.neurons:
                    if data[0] in self.network.neurons[post_neuron_id].incoming_synapses:
                        del self.network.neurons[post_neuron_id].incoming_synapses[data[0]]
        
        # Консолидация краткосрочной памяти
        if hasattr(self.network, 'memory_system'):
            consolidated_count = self.network.memory_system.consolidate_short_term_memories()
        
        logger.info(f"💤 Консолидация памяти завершена: +{strengthened} связей, "
                   f"-{weakened} связей, удалено {pruned} связей")
        
        return consolidated_count
    
    def optimize_synaptic_connections(self):
        """Оптимизация синаптических связей"""
        optimized_count = 0
        
        # Удаляем избыточные связи (слишком сильные и дублирующие)
        synapses_to_check = list(self.network.synapses.items())
        
        for syn_key, synapse in synapses_to_check:
            # Нормализуем слишком сильные связи
            if synapse.strength > 0.95:
                synapse.strength = 0.95
                optimized_count += 1
            
            # Удаляем связи между нейронами одного слоя (если это не рекуррентные связи)
            pre_neuron = self.network.neurons.get(synapse.pre_neuron_id)
            post_neuron = self.network.neurons.get(synapse.post_neuron_id)
            
            if (pre_neuron and post_neuron and 
                pre_neuron.layer_id == post_neuron.layer_id and
                random.random() < 0.1):  # 10% шанс удаления
                # Планируем для удаления
                self.memory_consolidation_queue.append(('prune_synapse', syn_key))
        
        return optimized_count
    
    def _restore_energy(self, rest_duration_minutes):
        """Восстановление уровня энергии"""
        # Базовое восстановление: 10% за 30 минут отдыха
        energy_restore_rate = (rest_duration_minutes / 30.0) * 0.1
        self.energy_level = min(1.0, self.energy_level + energy_restore_rate)
        
        # Дополнительное восстановление если были оптимизации
        if hasattr(self.network, 'synapses'):
            synapse_count = len(self.network.synapses)
            if synapse_count < 10000:  # Если сеть не перегружена
                self.energy_level = min(1.0, self.energy_level + 0.05)
    
    def get_rest_report(self):
        """Отчет о состоянии отдыха"""
        awake_duration = time.time() - self.awake_time
        
        return {
            'awake_duration_hours': awake_duration / 3600,
            'total_sleep_cycles': self.sleep_cycles,
            'total_sleep_time_minutes': self.sleep_duration / 60,
            'energy_level': self.energy_level,
            'needs_rest': self.needs_rest(),
            'consolidation_in_progress': self.consolidation_in_progress,
            'rest_history_count': len(self.rest_history)
        }

class VocabularyLearningSystem:
    """Система изучения лексики нейросети Sin"""
    
    def __init__(self, network):
        self.network = network
        self.wiktionary_base_url = "https://ru.wiktionary.org/w/api.php"
        self.learned_words = set()  # Слова, которые уже изучены
        self.learning_queue = deque(maxlen=1000)  # Очередь слов для изучения
        self.vocabulary_stats = {
            'total_learned': 0,
            'current_session': 0,
            'failed_lookups': 0
        }
        self.learning_active = False
        
        # Инициализируем парсеры
        if WIKTIONARY_PARSER_AVAILABLE:
            self.wiktionary_parser = WiktionaryParser()
        else:
            self.wiktionary_parser = None
    
    def get_word_definition(self, word):
        """Получение определения слова из Wiktionary"""
        try:
            # Попытка использовать wiktionaryparser
            if self.wiktionary_parser:
                try:
                    word_data = self.wiktionary_parser.fetch(word, "ru")
                    if word_data and word_data[0]:
                        definitions = []
                        examples = []
                        
                        for meaning in word_data[0]['definitions']:
                            definitions.append(meaning['definition'])
                        
                        for example_group in word_data[0].get('examples', []):
                            examples.extend(example_group)
                        
                        return {
                            'word': word,
                            'definitions': definitions[:3],  # Максимум 3 определения
                            'examples': examples[:5],  # Максимум 5 примеров
                            'source': 'Wiktionary (wiktionaryparser)'
                        }
                except Exception as e:
                    logger.warning(f"Ошибка wiktionaryparser для слова '{word}': {e}")
            
            # Резервный метод - ручной парсинг API
            params = {
                'action': 'parse',
                'page': word.lower(),
                'format': 'json',
                'prop': 'text'
            }
            
            response = requests.get(self.wiktionary_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                logger.warning(f"Ошибка API для слова '{word}': {data['error']['info']}")
                return None
            
            if 'parse' in data and 'text' in data['parse']:
                # Парсим HTML для извлечения определения
                html_content = data['parse']['text']['*']
                definition = self._parse_definition(html_content, word)
                return definition
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к Wiktionary для слова '{word}': {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка обработки данных для слова '{word}': {e}")
            return None
    
    def _parse_definition(self, html_content, word):
        """Парсинг HTML для извлечения определения"""
        if not BEAUTIFULSOUP_AVAILABLE:
            return None
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Ищем секцию с русским языком
            russian_section = None
            headers = soup.find_all(['h2', 'h3', 'h4'])
            
            for header in headers:
                if 'русский' in header.get_text().strip().lower():
                    russian_section = header
                    break
            
            if not russian_section:
                return None
            
            # Извлекаем определения
            definitions = []
            current_element = russian_section.find_next_sibling()
            
            # Ищем определения в различных форматах
            definition_keywords = ['значение', 'определение', 'значения']
            
            while current_element and current_element.name not in ['h2', 'h3']:
                text = current_element.get_text().strip()
                if (text and len(text) > 10 and 
                    any(keyword in text.lower() for keyword in definition_keywords)):
                    # Очищаем от лишних символов
                    clean_text = re.sub(r'\s+', ' ', text)
                    if clean_text and len(clean_text) > 15:
                        definitions.append(clean_text)
                
                current_element = current_element.find_next_sibling()
            
            # Извлекаем примеры использования
            examples = self._extract_examples(soup, russian_section)
            
            return {
                'word': word,
                'definitions': definitions[:3],  # Максимум 3 определения
                'examples': examples[:5],  # Максимум 5 примеров
                'source': 'Wiktionary'
            }
            
        except Exception as e:
            logger.error(f"Ошибка парсинга определения для '{word}': {e}")
            return None
    
    def _extract_examples(self, soup, start_section):
        """Извлечение примеров использования слова"""
        examples = []
        try:
            # Ищем примеры в различных секциях
            example_sections = soup.find_all(['dd', 'blockquote', 'li'])
            for section in example_sections:
                text = section.get_text().strip()
                if text and len(text) > 15:  # Минимальная длина примера
                    # Очищаем от лишних символов
                    clean_text = re.sub(r'\s+', ' ', text)
                    # Проверяем, что это действительно пример (содержит слово или похоже на предложение)
                    if (len(clean_text.split()) > 3 and 
                        (clean_text.endswith('.') or clean_text.endswith('!') or clean_text.endswith('?'))):
                        examples.append(clean_text)
        except Exception as e:
            logger.warning(f"Ошибка извлечения примеров: {e}")
        
        return examples[:5]  # Ограничиваем 5 примерами
    
    def add_words_to_learning_queue(self, words):
        """Добавление слов в очередь для изучения"""
        added_count = 0
        for word in words:
            word_lower = word.lower().strip()
            if word_lower and word_lower not in self.learned_words:
                self.learning_queue.append(word_lower)
                added_count += 1
        
        logger.info(f"Добавлено {added_count} слов в очередь изучения")
        return added_count
    
    def collect_words_from_sources(self):
        """Сбор слов из различных источников"""
        collected_words = set()
        
        # 1. Из токенизатора (новые слова)
        for token, freq in self.network.tokenizer.token_freq.items():
            if freq >= 2 and len(token) > 2:  # Только слова длиной > 2 и встречающиеся хотя бы 2 раза
                if re.match(r'^[а-яА-Я]+$', token):  # Только русские слова
                    collected_words.add(token.lower())
        
        # 2. Из истории разговоров
        for msg_type, msg_content, timestamp in list(self.network.conversation_history)[-50:]:  # Последние 50 сообщений
            if isinstance(msg_content, str):
                tokens = self.network.tokenizer.tokenize_basic(msg_content)
                for token in tokens:
                    if len(token) > 2 and re.match(r'^[а-яА-Я]+$', token):
                        collected_words.add(token.lower())
        
        # 3. Из памяти
        for memory_id, memory_data in list(self.network.memory_system.long_term_memory.items())[-20:]:
            if 'data' in memory_data:
                data_content = str(memory_data['data'])
                tokens = self.network.tokenizer.tokenize_basic(data_content)
                for token in tokens:
                    if len(token) > 2 and re.match(r'^[а-яА-Я]+$', token):
                        collected_words.add(token.lower())
        
        # 4. Из ранее изученных определений (новые слова из определений)
        for word in list(self.learned_words)[-10:]:  # Последние 10 изученных слов
            definition_data = self.get_word_definition(word)
            if definition_data:
                # Извлекаем новые слова из определений
                for definition in definition_data.get('definitions', []):
                    def_tokens = self.network.tokenizer.tokenize_basic(definition)
                    for token in def_tokens:
                        if len(token) > 2 and re.match(r'^[а-яА-Я]+$', token):
                            collected_words.add(token.lower())
                
                # Извлекаем новые слова из примеров
                for example in definition_data.get('examples', []):
                    example_tokens = self.network.tokenizer.tokenize_basic(example)
                    for token in example_tokens:
                        if len(token) > 2 and re.match(r'^[а-яА-Я]+$', token):
                            collected_words.add(token.lower())
        
        # Фильтруем уже изученные слова
        new_words = collected_words - self.learned_words
        
        logger.info(f"Собрано {len(new_words)} новых слов из различных источников")
        return list(new_words)
    
    def auto_learn_vocabulary(self, limit=50, delay=2.0):
        """Автоматическое изучение слов из очереди"""
        if self.learning_active:
            logger.warning("Изучение лексики уже запущено")
            return None
            
        self.learning_active = True
        logger.info(f"🚀 Начало автоматического изучения лексики (лимит: {limit} слов)")
        
        # Сначала собираем слова из источников
        logger.info("🔍 Сбор слов из различных источников...")
        collected_words = self.collect_words_from_sources()
        
        # Добавляем собранные слова в очередь
        if collected_words:
            self.add_words_to_learning_queue(collected_words)
            logger.info(f"📥 Добавлено {len(collected_words)} слов в очередь изучения")
        
        learned_count = 0
        failed_count = 0
        
        try:
            while learned_count < limit and self.learning_queue:
                if not self.learning_queue:
                    logger.info("Очередь слов для изучения пуста")
                    break
                
                word = self.learning_queue.popleft()
                
                if word in self.learned_words:
                    continue
                
                print(f"📖 Изучаю слово: {word}")
                
                # Получаем определение
                definition_data = self.get_word_definition(word)
                
                if definition_
                    # Обучаем нейросеть на новом слове
                    learning_result = self._train_on_word_definition(definition_data)
                    
                    if learning_result:
                        self.learned_words.add(word)
                        self.vocabulary_stats['total_learned'] += 1
                        self.vocabulary_stats['current_session'] += 1
                        learned_count += 1
                        
                        print(f"✅ Изучено слово: {word}")
                        print(f"   Определений: {len(definition_data.get('definitions', []))}")
                        print(f"   Примеров: {len(definition_data.get('examples', []))}")
                        
                        # Показываем первое определение
                        if definition_data.get('definitions'):
                            print(f"   Значение: {definition_data['definitions'][0][:100]}...")
                    else:
                        failed_count += 1
                        print(f"❌ Ошибка обучения на слове: {word}")
                else:
                    self.vocabulary_stats['failed_lookups'] += 1
                    failed_count += 1
                    print(f"❌ Не удалось найти определение для: {word}")
                
                # Пауза между запросами
                time.sleep(delay)
                
                # Показываем прогресс
                if learned_count % 10 == 0:
                    progress = (learned_count / limit) * 100
                    print(f"📊 Прогресс: {progress:.1f}% ({learned_count}/{limit})")
            
            logger.info(f"✅ Автоматическое изучение завершено!")
            logger.info(f"   Успешно изучено: {learned_count} слов")
            logger.info(f"   Ошибок: {failed_count}")
            logger.info(f"   Не найдено: {self.vocabulary_stats['failed_lookups']}")
            
            return {
                'learned': learned_count,
                'failed': failed_count,
                'total_in_queue': len(self.learning_queue)
            }
            
        finally:
            self.learning_active = False
    
    def _train_on_word_definition(self, definition_data):
        """Обучение нейросети на определении слова"""
        try:
            word = definition_data['word']
            definitions = definition_data.get('definitions', [])
            examples = definition_data.get('examples', [])
            
            # Создаем обучающие пары
            training_pairs = []
            
            # Пара: слово -> определение
            for definition in definitions[:2]:  # Максимум 2 определения
                training_pairs.append([f"Что означает слово {word}?", definition])
            
            # Пара: пример -> объяснение
            for example in examples[:3]:  # Максимум 3 примера
                training_pairs.append([f"Объясни пример: {example}", f"Это пример использования слова {word}"])
            
            # Обучаем нейросеть
            for pair in training_pairs:
                self.network.train_on_conversation(pair, is_new_info=True, user_id="vocabulary_system")
            
            # Также добавляем слово в токенизатор
            self.network.tokenizer.update_vocab([word] + definitions + examples, is_new_info=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения на определении слова: {e}")
            return False
    
    def get_vocabulary_report(self):
        """Отчет о состоянии изучения лексики"""
        return {
            'total_learned_words': self.vocabulary_stats['total_learned'],
            'session_learned': self.vocabulary_stats['current_session'],
            'failed_lookups': self.vocabulary_stats['failed_lookups'],
            'queue_size': len(self.learning_queue),
            'unique_words_studied': len(self.learned_words)
        }
    
    def search_word_meaning(self, word):
        """Поиск значения слова по запросу пользователя"""
        print(f"🔍 Ищу значение слова: {word}")
        
        definition_data = self.get_word_definition(word)
        
        if definition_
            response_lines = [f"📚 Слово: {definition_data['word'].upper()}"]
            
            if definition_data.get('definitions'):
                response_lines.append("\n📌 Определения:")
                for i, definition in enumerate(definition_data['definitions'][:3], 1):
                    response_lines.append(f"  {i}. {definition}")
            
            if definition_data.get('examples'):
                response_lines.append("\n📝 Примеры использования:")
                for i, example in enumerate(definition_data['examples'][:3], 1):
                    response_lines.append(f"  {i}. {example}")
            
            response_lines.append(f"\nℹ️ Источник: {definition_data['source']}")
            
            return "\n".join(response_lines)
        else:
            return f"❌ Не удалось найти определение для слова '{word}'"

class AdaptiveTokenizer:
    """Самообучающийся токенизатор нейросети Sin"""
    
    def __init__(self, vocab_file='sin_tokenizer_vocab.pkl'):
        self.vocab_file = vocab_file
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, '<SEP>': 4}
        self.vocab_size = 5
        self.token_freq = Counter()
        self.subword_freq = Counter()
        self.max_subword_length = 5
        self.bpe_pairs = {}
        self.embeddings = {}  # Простые эмбеддинги для токенов
        self.token_categories = {}  # Категории токенов
        self.load_vocab()
    
    def save_vocab(self):
        """Сохранение словаря"""
        try:
            with open(self.vocab_file, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'vocab_size': self.vocab_size,
                    'token_freq': self.token_freq,
                    'subword_freq': self.subword_freq,
                    'bpe_pairs': self.bpe_pairs,
                    'embeddings': self.embeddings,
                    'token_categories': self.token_categories
                }, f)
            logger.info(f"Токенизатор сохранен в {self.vocab_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения токенизатора: {e}")
    
    def load_vocab(self):
        """Загрузка словаря"""
        if os.path.exists(self.vocab_file):
            try:
                with open(self.vocab_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vocab = data.get('vocab', self.vocab)
                    self.vocab_size = data.get('vocab_size', self.vocab_size)
                    self.token_freq = data.get('token_freq', self.token_freq)
                    self.subword_freq = data.get('subword_freq', self.subword_freq)
                    self.bpe_pairs = data.get('bpe_pairs', self.bpe_pairs)
                    self.embeddings = data.get('embeddings', self.embeddings)
                    self.token_categories = data.get('token_categories', self.token_categories)
                logger.info(f"Токенизатор загружен из {self.vocab_file}")
            except Exception as e:
                logger.error(f"Ошибка загрузки токенизатора: {e}")
    
    def tokenize_basic(self, text):
        """Базовая токенизация"""
        if not isinstance(text, str):
            return []
        # Улучшенная токенизация
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return [token for token in tokens if token.strip()]
    
    def update_vocab(self, texts, is_new_info=False):
        """Обновление словаря на основе новых текстов"""
        new_tokens_count = 0
        processed_texts = 0
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
                
            tokens = self.tokenize_basic(text)
            self.token_freq.update(tokens)
            processed_texts += 1
            
            # Добавляем подслова для BPE и категоризируем токены
            for token in tokens:
                # Категоризация токенов
                self._categorize_token(token)
                
                # Подслова для BPE
                for i in range(len(token)):
                    for j in range(i+1, min(i+self.max_subword_length+1, len(token)+1)):
                        subword = token[i:j]
                        self.subword_freq[subword] += 1
                
                # Создаем простые эмбеддинги
                if token not in self.embeddings:
                    # Простой хеш-базированный эмбеддинг
                    hash_val = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
                    embedding = [((hash_val >> i) & 1) for i in range(32)]
                    self.embeddings[token] = embedding
        
        # Добавляем новые токены в словарь
        for token, freq in self.token_freq.most_common():
            if token not in self.vocab and freq > 1:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
                new_tokens_count += 1
        
        info_type = "новой" if is_new_info else "обучающей"
        if processed_texts > 0:
            logger.info(f"Обновлен словарь на основе {info_type} информации: "
                       f"+{new_tokens_count} токенов, обработано {processed_texts} текстов")
        return new_tokens_count
    
    def _categorize_token(self, token):
        """Категоризация токена"""
        if token.isdigit():
            self.token_categories[token] = 'number'
        elif re.match(r'^[a-zA-Z]+$', token):
            self.token_categories[token] = 'word'
        elif re.match(r'^[а-яА-Я]+$', token):
            self.token_categories[token] = 'russian_word'
        elif any(c in token for c in '.,!?;:'):
            self.token_categories[token] = 'punctuation'
        elif any(c in token for c in '+-*/=<>'):
            self.token_categories[token] = 'operator'
        else:
            self.token_categories[token] = 'other'
    
    def get_bpe_pairs(self, tokens):
        """Получение пар для BPE"""
        pairs = []
        for token in tokens:
            chars = list(token)
            for i in range(len(chars)-1):
                pairs.append((chars[i], chars[i+1]))
        return pairs
    
    def learn_bpe(self, texts, num_merges=5):
        """Обучение BPE"""
        vocab_changes = 0
        for merge_round in range(num_merges):
            pairs = Counter()
            for text in texts:
                tokens = self.tokenize_basic(text)
                token_pairs = self.get_bpe_pairs(tokens)
                pairs.update(token_pairs)
            
            if not pairs:
                break
                
            best_pair = pairs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]
            
            if new_token not in self.vocab and pairs[best_pair] > 3:
                self.vocab[new_token] = self.vocab_size
                self.vocab_size += 1
                self.bpe_pairs[best_pair] = new_token
                self.token_categories[new_token] = 'bpe_subword'
                vocab_changes += 1
                
                # Создаем эмбеддинг для нового подслова
                if best_pair[0] in self.embeddings and best_pair[1] in self.embeddings:
                    emb1 = self.embeddings[best_pair[0]]
                    emb2 = self.embeddings[best_pair[1]]
                    # Простое усреднение эмбеддингов
                    new_embedding = [(e1 + e2) / 2 for e1, e2 in zip(emb1, emb2)]
                    self.embeddings[new_token] = new_embedding
        
        if vocab_changes > 0:
            logger.info(f"BPE обучение: +{vocab_changes} новых подслов")
        return vocab_changes
    
    def encode(self, text):
        """Кодирование текста в токены"""
        if not isinstance(text, str):
            return [0]  # PAD token
            
        tokens = self.tokenize_basic(text)
        encoded = [self.vocab.get('<START>', 2)]
        
        for token in tokens:
            if token in self.vocab:
                encoded.append(self.vocab[token])
            else:
                # Ищем подслова
                found = False
                for subword in sorted(self.vocab.keys(), key=len, reverse=True):
                    if subword in token and len(subword) > 1:
                        encoded.append(self.vocab[subword])
                        found = True
                        break
                if not found:
                    encoded.append(self.vocab['<UNK>'])
        
        encoded.append(self.vocab.get('<END>', 3))
        return encoded[:512]  # Ограничиваем длину
    
    def decode(self, token_ids):
        """Декодирование токенов в текст"""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []
        
        for token_id in token_ids:
            if token_id in reverse_vocab:
                token = reverse_vocab[token_id]
                if token not in ['<PAD>', '<UNK>', '<START>', '<END>', '<SEP>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def get_vocab_report(self):
        """Отчет о состоянии словаря"""
        categories_count = Counter(self.token_categories.values())
        return {
            'vocab_size': self.vocab_size,
            'unique_tokens': len(self.token_freq),
            'total_token_frequency': sum(self.token_freq.values()),
            'token_categories': dict(categories_count),
            'bpe_pairs_count': len(self.bpe_pairs),
            'embeddings_count': len(self.embeddings)
        }

class DynamicNeuron:
    """Динамический нейрон с расширенной функциональностью"""
    
    def __init__(self, neuron_id, layer_id):
        self.id = neuron_id
        self.layer_id = layer_id
        self.activation = 0.0
        self.firing_threshold = random.uniform(0.3, 0.7)
        self.synaptic_strengths = {}  # {neuron_id: strength}
        self.incoming_synapses = {}
        self.learning_rate = 0.01
        self.adaptation_counter = 0
        self.importance_score = 0.0
        self.activation_history = deque(maxlen=200)
        self.is_active = False
        self.last_activation_time = 0
        self.neuron_type = self._determine_neuron_type()
        self.health_score = 1.0  # 0.0 - поврежден, 1.0 - здоров
    
    def _determine_neuron_type(self):
        """Определение типа нейрона"""
        if self.layer_id == 0:
            return 'input'
        elif self.layer_id == max(getattr(self, 'network_layers', [2])) - 1:
            return 'output'
        else:
            return 'hidden'
    
    def activate(self, inputs_dict):
        """Активация нейрона"""
        total_input = 0
        for source_id, signal in inputs_dict.items():
            if source_id in self.incoming_synapses:
                strength = self.incoming_synapses[source_id]
                total_input += signal * strength
        
        self.activation = self.sigmoid(total_input)
        self.is_active = self.activation > self.firing_threshold
        self.activation_history.append(self.activation)
        self.last_activation_time = time.time()
        
        # Адаптация порога
        if len(self.activation_history) > 20:
            avg_activation = sum(list(self.activation_history)[-20:]) / 20
            self._adapt_threshold(avg_activation)
        
        # Обновление важности
        self.update_importance()
        
        return self.activation
    
    def sigmoid(self, x):
        """Сигмоидальная функция"""
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1 / (1 + np.exp(-x))
    
    def _adapt_threshold(self, avg_activation):
        """Адаптация порога активации"""
        if avg_activation > 0.8:
            self.firing_threshold = min(0.9, self.firing_threshold * 1.03)
        elif avg_activation < 0.2:
            self.firing_threshold = max(0.1, self.firing_threshold * 0.97)
    
    def update_importance(self):
        """Обновление важности нейрона"""
        if len(self.activation_history) > 50:
            recent_activity = sum(list(self.activation_history)[-50:]) / 50
            # Также учитываем вариативность активации
            if len(self.activation_history) > 100:
                std_dev = np.std(list(self.activation_history)[-50:])
                self.importance_score = (recent_activity * 0.7 + std_dev * 0.3)
            else:
                self.importance_score = recent_activity
    
    def get_neuron_report(self):
        """Отчет о состоянии нейрона"""
        recent_activations = list(self.activation_history)[-20:] if self.activation_history else [0]
        return {
            'id': self.id,
            'layer_id': self.layer_id,
            'type': self.neuron_type,
            'current_activation': self.activation,
            'is_active': self.is_active,
            'firing_threshold': self.firing_threshold,
            'importance_score': self.importance_score,
            'recent_activity': sum(recent_activations) / max(1, len(recent_activations)) if recent_activations else 0,
            'synapse_count': len(self.incoming_synapses),
            'health_score': self.health_score
        }

class DynamicSynapse:
    """Динамический синапс с пластичностью"""
    
    def __init__(self, pre_neuron_id, post_neuron_id):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.strength = random.uniform(0.1, 0.5)
        self.ltp_counter = 0
        self.ltd_counter = 0
        self.is_active = True
        self.creation_time = time.time()
        self.last_update_time = time.time()
        self.efficiency = 1.0  # Эффективность передачи сигнала
    
    def update_strength(self, delta):
        """Обновление силы синапса"""
        old_strength = self.strength
        self.strength = np.clip(self.strength + delta, -1.0, 1.0)
        self.last_update_time = time.time()
        
        # Обновляем эффективность на основе изменений
        change_magnitude = abs(self.strength - old_strength)
        if change_magnitude > 0.1:
            self.efficiency = min(1.0, self.efficiency + 0.05)
        elif change_magnitude < 0.01:
            self.efficiency = max(0.1, self.efficiency - 0.01)
    
    def apply_plasticity(self, pre_active, post_active):
        """Применение нейропластичности"""
        self.last_update_time = time.time()
        
        if pre_active and post_active:
            # Long Term Potentiation
            self.ltp_counter += 1
            if self.ltp_counter > 4:
                self.strength = min(1.0, self.strength * 1.12)
                self.ltp_counter = max(0, self.ltp_counter - 2)
                self.ltd_counter = max(0, self.ltd_counter - 1)
        elif pre_active and not post_active:
            # Long Term Depression
            self.ltd_counter += 1
            if self.ltd_counter > 10:
                self.strength = max(-1.0, self.strength * 0.93)
                self.ltd_counter = max(0, self.ltd_counter - 3)
                self.ltp_counter = max(0, self.ltp_counter - 1)
        elif not pre_active and post_active:
            # Слабая депрессия
            self.ltd_counter += 0.5
            if self.ltd_counter > 15:
                self.strength = max(-1.0, self.strength * 0.97)
    
    def get_synapse_report(self):
        """Отчет о состоянии синапса"""
        age_hours = (time.time() - self.creation_time) / 3600
        return {
            'pre_neuron_id': self.pre_neuron_id,
            'post_neuron_id': self.post_neuron_id,
            'strength': self.strength,
            'ltp_counter': self.ltp_counter,
            'ltd_counter': self.ltd_counter,
            'efficiency': self.efficiency,
            'age_hours': age_hours,
            'is_active': self.is_active
        }

class SinNeuralNetwork:
    """Главная нейросеть Sin с полной интеграцией всех систем"""
    
    def __init__(self, model_file='sin_model.pkl'):
        self.model_file = model_file
        self.tokenizer = AdaptiveTokenizer()
        
        # Основные компоненты архитектуры
        self.layers = {}
        self.neurons = {}
        self.synapses = {}
        self.neuron_counter = 0
        self.layer_counter = 0
        self.max_layers = 20
        
        # История и статистика
        self.conversation_history = deque(maxlen=1000)
        self.learning_sessions = 0
        self.performance_history = deque(maxlen=1000)
        self.context_memory = deque(maxlen=20)
        
        # Активные нейроны и статистика
        self.active_neurons = 0
        self.total_neurons = 0
        self.system_stats = {
            'cpu_percent': 0,
            'memory_percent': 0,
            'network_speed': 'normal'
        }
        
        # Интегрированные системы
        self.emotional_system = EmotionalSystem()
        self.attention_system = AttentionSystem(self)
        self.memory_system = MemorySystem()
        self.metacognition_system = MetacognitionSystem(self)
        self.social_system = SocialInteractionSystem()
        self.creativity_system = CreativitySystem(self)
        self.sleep_system = SleepAndRestSystem(self)
        self.vocabulary_system = VocabularyLearningSystem(self)
        
        # Адаптивные параметры обучения
        self.adaptive_params = self._get_adaptive_params()
        
        # Инициализируем сеть
        self.initialize_network()
        
        logger.info("🧠 Нейросеть Sin инициализирована")
    
    def _get_adaptive_params(self):
        """Определение параметров обучения под мощность железа"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Базовые параметры
        params = {
            'batch_size': min(64, max(16, cpu_count * 4)),
            'learning_rate': 0.015,
            'epochs_per_session': 8,
            'max_new_neurons_per_session': 150,
            'synapse_density': 0.35,
            'pruning_threshold': 0.1
        }
        
        # Адаптация под ресурсы
        if memory_gb < 8:
            params['batch_size'] = 16
            params['max_new_neurons_per_session'] = 75
            params['synapse_density'] = 0.25
        elif memory_gb > 32:
            params['batch_size'] = 128
            params['max_new_neurons_per_session'] = 300
            params['synapse_density'] = 0.45
        
        logger.info(f"⚙️ Адаптивные параметры: {params}")
        return params
    
    def _create_initial_architecture(self):
        """Создание начальной архитектуры"""
        # Входной слой (150 нейронов)
        input_layer = []
        for i in range(150):
            neuron = DynamicNeuron(self.neuron_counter, 0)
            self.neurons[self.neuron_counter] = neuron
            input_layer.append(self.neuron_counter)
            self.neuron_counter += 1
        self.layers[0] = input_layer
        
        # Первый скрытый слой (100 нейронов)
        hidden_layer_1 = []
        for i in range(100):
            neuron = DynamicNeuron(self.neuron_counter, 1)
            self.neurons[self.neuron_counter] = neuron
            hidden_layer_1.append(self.neuron_counter)
            self.neuron_counter += 1
        self.layers[1] = hidden_layer_1
        
        # Второй скрытый слой (75 нейронов)
        hidden_layer_2 = []
        for i in range(75):
            neuron = DynamicNeuron(self.neuron_counter, 2)
            self.neurons[self.neuron_counter] = neuron
            hidden_layer_2.append(self.neuron_counter)
            self.neuron_counter += 1
        self.layers[2] = hidden_layer_2
        
        # Выходной слой (150 нейронов)
        output_layer = []
        for i in range(150):
            neuron = DynamicNeuron(self.neuron_counter, 3)
            self.neurons[self.neuron_counter] = neuron
            output_layer.append(self.neuron_counter)
            self.neuron_counter += 1
        self.layers[3] = output_layer
        
        # Создаем начальные связи
        self._create_initial_connections()
        self.layer_counter = 4
        self.total_neurons = len(self.neurons)
        self._update_active_neurons()
        
        logger.info("🏗️ Создана начальная архитектура сети")
    
    def _create_initial_connections(self):
        """Создание начальных синаптических связей"""
        density = self.adaptive_params['synapse_density']
        
        # Связи между входным и первым скрытым слоем
        for input_neuron_id in self.layers[0]:
            for hidden_neuron_id in self.layers[1]:
                if random.random() < density:
                    synapse = DynamicSynapse(input_neuron_id, hidden_neuron_id)
                    self.synapses[(input_neuron_id, hidden_neuron_id)] = synapse
                    self.neurons[hidden_neuron_id].incoming_synapses[input_neuron_id] = synapse.strength
        
        # Связи между первым и вторым скрытым слоем
        for hidden1_neuron_id in self.layers[1]:
            for hidden2_neuron_id in self.layers[2]:
                if random.random() < density * 0.8:
                    synapse = DynamicSynapse(hidden1_neuron_id, hidden2_neuron_id)
                    self.synapses[(hidden1_neuron_id, hidden2_neuron_id)] = synapse
                    self.neurons[hidden2_neuron_id].incoming_synapses[hidden1_neuron_id] = synapse.strength
        
        # Связи между вторым скрытым и выходным слоем
        for hidden2_neuron_id in self.layers[2]:
            for output_neuron_id in self.layers[3]:
                if random.random() < density:
                    synapse = DynamicSynapse(hidden2_neuron_id, output_neuron_id)
                    self.synapses[(hidden2_neuron_id, output_neuron_id)] = synapse
                    self.neurons[output_neuron_id].incoming_synapses[hidden2_neuron_id] = synapse.strength
    
    def _update_system_stats(self):
        """Обновление системной статистики"""
        self.system_stats['cpu_percent'] = psutil.cpu_percent()
        self.system_stats['memory_percent'] = psutil.virtual_memory().percent
    
    def _add_new_layer(self):
        """Добавление нового слоя"""
        if len(self.layers) >= self.max_layers:
            logger.info("Достигнуто максимальное количество слоев")
            return False
            
        new_layer_id = self.layer_counter
        previous_layer_id = new_layer_id - 1
        
        # Создаем нейроны нового слоя
        new_layer = []
        prev_layer_size = len(self.layers[previous_layer_id])
        new_layer_size = max(15, prev_layer_size // 2)
        
        for i in range(new_layer_size):
            neuron = DynamicNeuron(self.neuron_counter, new_layer_id)
            # Передаем информацию о слоях для правильной типизации
            neuron.network_layers = list(self.layers.keys())
            self.neurons[self.neuron_counter] = neuron
            new_layer.append(self.neuron_counter)
            self.neuron_counter += 1
        
        self.layers[new_layer_id] = new_layer
        
        # Создаем связи с предыдущим слоем
        density = self.adaptive_params['synapse_density'] * 0.6
        for prev_neuron_id in self.layers[previous_layer_id]:
            for new_neuron_id in new_layer:
                if random.random() < density:
                    synapse = DynamicSynapse(prev_neuron_id, new_neuron_id)
                    self.synapses[(prev_neuron_id, new_neuron_id)] = synapse
                    self.neurons[new_neuron_id].incoming_synapses[prev_neuron_id] = synapse.strength
        
        # Пересоздаем связи с выходным слоем (если нужно)
        if new_layer_id < max(self.layers.keys()) - 1:
            self._reconnect_layers(previous_layer_id, new_layer_id)
        
        self.layer_counter += 1
        self.total_neurons = len(self.neurons)
        self._update_active_neurons()
        
        logger.info(f"➕ Добавлен новый слой {new_layer_id} с {len(new_layer)} нейронами")
        return True
    
    def _reconnect_layers(self, old_prev_layer_id, new_layer_id):
        """Пересоединение слоев при добавлении нового"""
        next_layer_id = new_layer_id + 1
        if next_layer_id in self.layers:
            # Удаляем старые связи
            connections_to_remove = []
            for syn_key in self.synapses:
                pre_id, post_id = syn_key
                if (pre_id in self.layers[old_prev_layer_id] and 
                    post_id in self.layers[next_layer_id]):
                    connections_to_remove.append(syn_key)
            
            for syn_key in connections_to_remove:
                del self.synapses[syn_key]
                if post_id in self.neurons:
                    if syn_key[0] in self.neurons[post_id].incoming_synapses:
                        del self.neurons[post_id].incoming_synapses[syn_key[0]]
            
            # Создаем новые связи через новый слой
            density = self.adaptive_params['synapse_density'] * 0.4
            for new_neuron_id in self.layers[new_layer_id]:
                for next_neuron_id in self.layers[next_layer_id]:
                    if random.random() < density:
                        synapse = DynamicSynapse(new_neuron_id, next_neuron_id)
                        self.synapses[(new_neuron_id, next_neuron_id)] = synapse
                        self.neurons[next_neuron_id].incoming_synapses[new_neuron_id] = synapse.strength
    
    def _update_active_neurons(self):
        """Обновление списка активных нейронов"""
        active_count = 0
        current_time = time.time()
        
        for neuron in self.neurons.values():
            if current_time - neuron.last_activation_time < 120:  # Активные за последние 2 минуты
                active_count += 1
        
        self.active_neurons = active_count
    
    def forward(self, input_tokens):
        """Прямой проход через сеть"""
        activations = {}
        
        # Заполняем входной слой
        input_layer = self.layers[0]
        for i, token in enumerate(input_tokens[:len(input_layer)]):
            if i < len(input_layer):
                neuron_id = input_layer[i]
                normalized_input = token / max(1, self.tokenizer.vocab_size)
                self.neurons[neuron_id].activation = normalized_input
                activations[neuron_id] = normalized_input
        
        # Активация остальных слоев
        for layer_id in sorted(self.layers.keys()):
            if layer_id == 0:
                continue
                
            layer_neurons = self.layers[layer_id]
            layer_activations = {}
            
            for neuron_id in layer_neurons:
                inputs_dict = {}
                for syn_key, synapse in self.synapses.items():
                    if synapse.post_neuron_id == neuron_id:
                        pre_neuron_id = synapse.pre_neuron_id
                        if pre_neuron_id in activations:
                            # Применяем эффективность синапса
                            effective_signal = activations[pre_neuron_id] * synapse.efficiency
                            inputs_dict[pre_neuron_id] = effective_signal
                
                activation = self.neurons[neuron_id].activate(inputs_dict)
                layer_activations[neuron_id] = activation
                activations[neuron_id] = activation
            
            # Обновляем активные нейроны
            self._update_active_neurons()
        
        # Возвращаем активации выходного слоя
        output_layer = self.layers[max(self.layers.keys())]
        output_activations = [activations.get(neuron_id, 0.0) for neuron_id in output_layer]
        
        return output_activations
    
    def generate_response(self, input_text, max_length=40, temperature=1.0, context_aware=True):
        """Генерация ответа с учетом всех систем"""
        # Сохраняем контекст
        if context_aware:
            self.context_memory.append(input_text)
        
        # Объединяем контекст с текущим запросом
        if context_aware and len(self.context_memory) > 1:
            context_text = " ".join(list(self.context_memory)[-3:])  # Последние 3 сообщения
            full_input = context_text + " |ТЕКУЩИЙ ЗАПРОС: " + input_text
        else:
            full_input = input_text
        
        # Получаем веса внимания
        input_tokens = self.tokenizer.encode(full_input)
        context_tokens = self.tokenizer.encode(context_text) if context_aware and len(self.context_memory) > 1 else []
        attention_weights = self.attention_system.calculate_attention_weights(input_tokens, context_tokens)
        
        # Применяем влияние эмоций
        emotion_influence = self.emotional_system.influence_on_responses()
        adjusted_temperature = temperature * emotion_influence.get('temperature', 1.0)
        creativity_boost = emotion_influence.get('creativity', 1.0)
        
        # Проверяем необходимость креативного ответа
        needs_creativity = (
            self.emotional_system.emotions['curiosity'] > 0.7 or
            self.emotional_system.emotions['excitement'] > 0.6 or
            creativity_boost > 1.2
        )
        
        if needs_creativity and random.random() < 0.3:
            # Генерируем креативный ответ
            response = self.creativity_system.generate_creative_response(
                full_input, 
                temperature=adjusted_temperature,
                max_attempts=2
            )
        else:
            # Стандартная генерация
            response_tokens = []
            
            # Генерируем ответ по токенам
            for gen_step in range(max_length):
                # Добавляем веса внимания к входу
                weighted_input = []
                for i, (token, weight) in enumerate(zip(input_tokens + response_tokens, 
                                                       attention_weights + [1.0] * len(response_tokens))):
                    # Применяем вес внимания
                    weighted_value = int(token * weight) if isinstance(token, (int, float)) else token
                    weighted_input.append(weighted_value)
                
                output_activations = self.forward(weighted_input[:512])  # Ограничиваем длину
                
                if output_activations:
                    # Нормализуем активации
                    activations_sum = sum(abs(a) for a in output_activations)
                    if activations_sum > 0:
                        normalized = [a/activations_sum for a in output_activations]
                    else:
                        normalized = [1/len(output_activations)] * len(output_activations)
                    
                    # Применяем температуру
                    if adjusted_temperature != 1.0:
                        normalized = [max(0, p/adjusted_temperature) for p in normalized]
                        norm_sum = sum(normalized)
                        if norm_sum > 0:
                            normalized = [p/norm_sum for p in normalized]
                    
                    # Выбираем токен
                    try:
                        next_token_idx = np.random.choice(len(normalized), p=normalized)
                        next_token_id = next_token_idx
                        
                        # Проверяем на специальные токены
                        if next_token_id == self.tokenizer.vocab.get('<END>', 3):
                            break
                        
                        response_tokens.append(next_token_id)
                    except:
                        break
                else:
                    break
            
            # Декодируем ответ
            response = self.tokenizer.decode(response_tokens)
        
        # Постобработка ответа с учетом социальных предпочтений
        processed_response = self._postprocess_response(response, input_text)
        
        return processed_response if processed_response.strip() else "Интересный вопрос! Можете рассказать подробнее?"
    
    def _postprocess_response(self, response, input_text):
        """Постобработка ответа с учетом социальных и эмоциональных факторов"""
        # Адаптируем стиль ответа под пользователя
        # Пока используем упрощенную версию
        if len(response) > 200:  # Ограничиваем длину
            response = ' '.join(response.split()[:50]) + "..."
        
        # Добавляем эмоциональную окраску если нужно
        if self.emotional_system.emotional_state == 'happy':
            positive_endings = ["!", " 😊", ". Интересно!", " 🌟"]
            if response and not any(response.endswith(ending) for ending in ['!', '?', '.']):
                response += random.choice(positive_endings)
        
        return response.strip()
    
    def _calculate_information_value(self, text):
        """Расчет ценности информации"""
        tokens = self.tokenizer.tokenize_basic(text)
        new_tokens = 0
        familiar_tokens = 0
        
        for token in tokens:
            if token in self.tokenizer.vocab:
                familiar_tokens += 1
            else:
                new_tokens += 1
        
        # Оценка новизны (0-1)
        novelty_score = new_tokens / max(1, len(tokens))
        
        # Оценка полезности (комплексная метрика)
        usefulness_score = min(1.0, (new_tokens * 0.7 + familiar_tokens * 0.3) / max(1, len(tokens)))
        
        return {
            'novelty': novelty_score,
            'usefulness': usefulness_score,
            'new_tokens': new_tokens,
            'familiar_tokens': familiar_tokens,
            'total_tokens': len(tokens)
        }
    
    def train_on_conversation(self, conversation_pair, is_new_info=False, user_id="default"):
        """Обучение на одной паре диалога"""
        if len(conversation_pair) < 2:
            return 0.0
            
        user_msg, bot_response = conversation_pair[0], conversation_pair[1]
        
        # Анализируем информацию
        info_metrics = self._calculate_information_value(user_msg + " " + bot_response)
        
        # Обновляем токенизатор
        new_tokens = self.tokenizer.update_vocab([user_msg, bot_response], is_new_info)
        
        # Обучаем BPE
        if new_tokens > 0:
            self.tokenizer.learn_bpe([user_msg, bot_response], num_merges=2)
        
        # Симуляция обучения
        input_tokens = self.tokenizer.encode(user_msg)
        target_tokens = self.tokenizer.encode(bot_response)
        
        learning_result = self._simulate_learning(input_tokens, target_tokens)
        
        # Обновляем системы
        self.emotional_system.update_emotions({
            'novelty': info_metrics['novelty'],
            'accuracy': learning_result.get('accuracy', 0.5),
            'repetition': False,
            'topic_interest': 0.5
        })
        
        # Обновляем внимание
        topics = self._extract_topics(user_msg)
        for topic in topics:
            self.attention_system.shift_attention(topic, info_metrics['novelty'])
        
        # Сохраняем в память
        memory_importance = (info_metrics['novelty'] * 0.6 + info_metrics['usefulness'] * 0.4)
        self.memory_system.store_memory(
            {'user_input': user_msg, 'bot_response': bot_response},
            importance=memory_importance,
            context={'topics': topics, 'novelty': info_metrics['novelty']}
        )
        
        # Обновляем социальный профиль
        self.social_system.create_user_profile(user_id, {
            'message': user_msg,
            'response': bot_response,
            'response_quality': learning_result.get('accuracy', 0.5),
            'feedback': 0  # Нейтральная обратная связь
        })
        
        # Обновляем метапознание
        for topic in topics:
            self.metacognition_system.assess_knowledge(topic)
        
        # Логируем результат
        log_msg = f"🎓 Обучение: "
        if is_new_info:
            log_msg += "НОВАЯ ИНФОРМАЦИЯ - "
        log_msg += f"Новизна: {info_metrics['novelty']:.2f}, "
        log_msg += f"Полезность: {info_metrics['usefulness']:.2f}, "
        log_msg += f"Новые токены: {info_metrics['new_tokens']}"
        
        logger.info(log_msg)
        
        return info_metrics['usefulness']
    
    def _extract_topics(self, text):
        """Извлечение тем из текста"""
        # Простая реализация - можно улучшить
        words = text.lower().split()
        # Базовые категории тем
        topic_keywords = {
            'technology': ['компьютер', 'ai', 'программ', 'робот', 'интернет', 'технолог'],
            'science': ['наука', 'физик', 'хими', 'биолог', 'математ', 'астроном'],
            'philosophy': ['мысл', 'смысл', 'реальн', 'существован', 'философ'],
            'literature': ['книг', 'поэз', 'роман', 'автор', 'литератур'],
            'history': ['истор', 'прошл', 'век', 'год', 'эпох']
        }
        
        detected_topics = set()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                detected_topics.add(topic)
        
        return list(detected_topics) if detected_topics else ['general']
    
    def _simulate_learning(self, input_tokens, target_tokens):
        """Симуляция процесса обучения"""
        output_activations = self.forward(input_tokens)
        
        output_layer = self.layers[max(self.layers.keys())]
        
        accuracy = 0.0
        processed_synapses = 0
        
        for i, (neuron_id, activation) in enumerate(zip(output_layer, output_activations)):
            target_activation = 0.9 if i < len(target_tokens) else 0.1
            error = target_activation - activation
            
            # Обновляем входящие синапсы
            for syn_key, synapse in self.synapses.items():
                if synapse.post_neuron_id == neuron_id:
                    delta = 0.008 * error * self.neurons[synapse.pre_neuron_id].activation
                    synapse.update_strength(delta)
                    self.neurons[neuron_id].incoming_synapses[synapse.pre_neuron_id] = synapse.strength
                    processed_synapses += 1
        
        # Оценка точности
        if len(target_tokens) > 0 and len(output_activations) > 0:
            # Простая оценка: чем ближе активации к целевым значениям, тем выше точность
            target_matches = min(len(target_tokens), len(output_activations))
            if target_matches > 0:
                accuracy = max(0.0, 1.0 - abs(len(target_tokens) - len(output_activations)) / max(1, len(target_tokens)))
        
        self.performance_history.append(accuracy)
        
        return {
            'accuracy': accuracy,
            'synapses_updated': processed_synapses,
            'learning_rate': self.adaptive_params['learning_rate']
        }
    
    def chat(self, user_input, user_id="default"):
        """Основной метод для чата"""
        conversation_start = time.time()
        
        # Сохраняем в историю
        self.conversation_history.append(("user", user_input, time.time()))
        
        # Генерируем ответ
        response = self.generate_response(user_input)
        
        # Сохраняем ответ в историю
        self.conversation_history.append(("bot", response, time.time()))
        
        # Обновляем системы
        response_time = time.time() - conversation_start
        
        # Обновляем эмоциональную систему
        self.emotional_system.update_emotions({
            'novelty': 0.3,  # Средняя новизна
            'accuracy': 0.7,  # Предполагаем среднюю точность
            'repetition': user_input in [msg[1] for msg in list(self.conversation_history)[-10:] if msg[0] == "user"],
            'topic_interest': 0.5
        })
        
        # Обновляем социальную систему
        self.social_system.create_user_profile(user_id, {
            'message': user_input,
            'response': response,
            'response_quality': len(response) / max(1, len(user_input)),  # Простая метрика качества
            'feedback': 0  # Нейтральная обратная связь
        })
        
        # Проверяем необходимость отдыха
        if self.sleep_system.needs_rest():
            logger.info("💤 Система нуждается в отдыхе")
            # В реальной системе здесь можно запланировать отдых
        
        # Обновляем внимание
        topics = self._extract_topics(user_input)
        for topic in topics:
            self.attention_system.shift_attention(topic)
        
        # Самоанализ
        if len(self.conversation_history) % 20 == 0:  # Каждые 10 диалогов
            self.metacognition_system.self_reflect()
        
        return response
    
    def save_model(self):
        """Сохранение модели"""
        try:
            model_data = {
                'layers': self.layers,
                'neuron_counter': self.neuron_counter,
                'layer_counter': self.layer_counter,
                'synapses': {str(k): v.__dict__ for k, v in self.synapses.items()},
                'neurons': {k: {**v.__dict__, 'activation_history': list(v.activation_history)} for k, v in self.neurons.items()},
                'conversation_history': list(self.conversation_history),
                'learning_sessions': self.learning_sessions,
                'performance_history': list(self.performance_history),
                'context_memory': list(self.context_memory)
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Сохраняем все подсистемы
            self.tokenizer.save_vocab()
            
            logger.info(f"💾 Модель сохранена в {self.model_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    def load_model(self):
        """Загрузка модели"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.layers = model_data['layers']
                self.neuron_counter = model_data['neuron_counter']
                self.layer_counter = model_data['layer_counter']
                self.conversation_history = deque(model_data['conversation_history'], maxlen=1000)
                self.learning_sessions = model_data['learning_sessions']
                self.performance_history = deque(model_data['performance_history'], maxlen=1000)
                self.context_memory = deque(model_data['context_memory'], maxlen=20)
                
                # Восстанавливаем нейроны
                restored_neurons = {}
                for neuron_id, neuron_data in model_data['neurons'].items():
                    neuron = DynamicNeuron(neuron_id, neuron_data['layer_id'])
                    # Восстанавливаем историю активаций
                    if 'activation_history' in neuron_
                        neuron.activation_history = deque(neuron_data['activation_history'], maxlen=200)
                        del neuron_data['activation_history']
                    neuron.__dict__.update(neuron_data)
                    restored_neurons[neuron_id] = neuron
                self.neurons = restored_neurons
                
                # Восстанавливаем синапсы
                restored_synapses = {}
                for syn_key_str, syn_data in model_data['synapses'].items():
                    syn_key = eval(syn_key_str)
                    synapse = DynamicSynapse(syn_key[0], syn_key[1])
                    synapse.__dict__.update(syn_data)
                    restored_synapses[syn_key] = synapse
                self.synapses = restored_synapses
                
                self.total_neurons = len(self.neurons)
                self._update_active_neurons()
                
                logger.info(f"📥 Модель загружена из {self.model_file}")
                logger.info(f"📊 Статистика: {self.get_stats()}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели: {e}")
        else:
            # Если модель не найдена, создаем новую архитектуру
            logger.warning(f"⚠️ Модель не найдена в {self.model_file}. Создание новой архитектуры.")
            self._create_initial_architecture()
    
    def initialize_network(self):
        """Инициализация нейросети"""
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
        
        if not hasattr(self, 'layers') or not self.layers:
            # Если модель не загрузилась, создаем новую архитектуру
            logger.info("⚠️ Создана новая архитектура")
            self._create_initial_architecture()
    
    def get_stats(self):
        """Получение статистики сети"""
        return {
            "layers": len(self.layers),
            "total_neurons": self.total_neurons,
            "active_neurons": self.active_neurons,
            "synapses": len(self.synapses),
            "vocab_size": self.tokenizer.vocab_size,
            "learning_sessions": self.learning_sessions,
            "conversation_history": len(self.conversation_history),
            "context_length": len(self.context_memory),
            "performance": sum(self.performance_history) / max(1, len(self.performance_history)) if self.performance_history else 0.0
        }
    
    def get_comprehensive_report(self):
        """Полный отчет о состоянии всех систем"""
        return {
            'network_stats': self.get_stats(),
            'emotional_state': self.emotional_system.get_emotional_report(),
            'attention_state': self.attention_system.get_attention_report(),
            'memory_state': self.memory_system.get_memory_report(),
            'metacognition_state': self.metacognition_system.get_metacognition_report(),
            'social_state': self.social_system.get_social_report(),
            'creativity_state': self.creativity_system.get_creativity_report(),
            'rest_state': self.sleep_system.get_rest_report(),
            'vocab_state': self.tokenizer.get_vocab_report(),
            'vocabulary_learning_state': self.vocabulary_system.get_vocabulary_report()
        }

# Основной класс консоли для Sin
class SinConsole:
    """Консольный интерфейс для нейросети Sin"""
    
    def __init__(self):
        self.setup_directories()
        self.network = SinNeuralNetwork()
        self.current_user_id = "default_user"
        self.session_start_time = time.time()
        self.auto_learning_active = False
        self.vocabulary_learning_active = False
    
    def setup_directories(self):
        """Создание необходимых директорий"""
        dirs = ['models', 'data', 'logs', 'datasets', 'sessions']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
        logger.info("📁 Директории созданы/проверены")
    
    def show_welcome(self):
        """Приветственное сообщение"""
        print("=" * 70)
        print("🤖 SIN - Самосознающая Интеллектуальная Нейросеть")
        print("=" * 70)
        print("🧠 Версия: 1.0 | 🧬 Состояние: Активна")
        print("\nКоманды:")
        print("  /chat <сообщение>    - Обычный чат")
        print("  /learn <текст>       - Обучение на тексте")
        print("  /file <путь>         - Обучение на файле")
        print("  /url <ссылка>        - Обучение по URL")
        print("  /auto <минуты>       - Автообучение (генерация текстов)")
        print("  /define <слово>      - Поиск значения слова")
        print("  /add_words <слова>   - Добавить слова для изучения")
        print("  /learn_words <лимит> - Автоизучение слов (по умолчанию 20)")
        print("  /vocab_stats         - Статистика изучения лексики")
        print("  /stats               - Полная статистика")
        print("  /report              - Комплексный отчет")
        print("  /save                - Сохранить модель")
        print("  /load                - Загрузить модель")
        print("  /user <id>           - Сменить пользователя")
        print("  /FAQ                 - Часто задаваемые вопросы")
        print("  /help                - Помощь")
        print("  /exit                - Выход")
        print("=" * 70)
    
    def show_faq(self):
        """Показ FAQ - часто задаваемых вопросов"""
        faq_text = """
📖 ЧАСТО ЗАДАВАЕМЫЕ ВОПРОСЫ (FAQ)

🔹 ОСНОВНЫЕ КОМАНДЫ:
  /chat <сообщение> - Обычный диалог с Sin
    Пример: /chat Привет, как дела?

  /learn <текст> - Обучение на введенном тексте
    Пример: /learn Искусственный интеллект - это область компьютерных наук...

  /file <путь> - Обучение на содержимом файла
    Поддерживаемые форматы: .txt, .docx, .pdf, .json
    Пример: /file /home/user/document.txt

  /url <ссылка> - Обучение на содержимом веб-страницы
    Пример: /url https://ru.wikipedia.org/wiki/Искусственный_интеллект

  /auto <минуты> - Автоматическое обучение (генерация текстов)
    Пример: /auto 30 (автообучение 30 минут)

🔹 РАБОТА С ЛЕКСИКОЙ:
  /define <слово> - Поиск значения слова в Wiktionary
    Пример: /define философия

  /add_words <слова> - Добавление слов в очередь изучения
    Пример: /add_words любовь, ненависть, счастье, грусть

  /learn_words <лимит> - Автоматическое изучение слов из очереди
    Пример: /learn_words 50 (изучить 50 слов)

  /vocab_stats - Статистика изучения лексики
    Показывает количество изученных слов, ошибок и т.д.

🔹 СИСТЕМНЫЕ КОМАНДЫ:
  /stats - Общая статистика нейросети
    Показывает количество нейронов, синапсов, сессий обучения

  /report - Комплексный отчет о состоянии всех систем
    Подробная информация о всех компонентах Sin

  /save - Сохранение текущего состояния модели
    Модель сохраняется в файл sin_model.pkl

  /load - Загрузка сохраненной модели
    Загружает модель из файла sin_model.pkl

  /user <id> - Смена пользователя (для персонализации)
    Пример: /user user123

  /FAQ - Показ этой справки
  /help - Показ списка команд

🔹 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:
  1. Начните с простого общения: /chat Привет!
  2. Для обучения используйте качественные тексты
  3. Не перегружайте систему большим объемом данных за раз
  4. Регулярно сохраняйте модель командой /save
  5. Используйте /learn_words для расширения словарного запаса
  6. Проверяйте статистику командой /stats

🔹 ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ:
  - Модель сохраняется автоматически при выходе
  - Поддерживается автоматическое определение кодировки файлов
  - Для работы с Wiktionary требуется интернет-соединение
  - Система адаптирует параметры под мощность вашего компьютера
"""
        print(faq_text)
        return True
    
    def process_command(self, user_input):
        """Обработка команд"""
        if user_input.startswith('/'):
            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            command_methods = {
                '/chat': self._chat_command,
                '/learn': self._learn_command,
                '/file': self._file_command,
                '/url': self._url_command,
                '/auto': self._auto_command,
                '/stats': self._stats_command,
                '/report': self._report_command,
                '/save': self._save_command,
                '/load': self._load_command,
                '/user': self._user_command,
                '/help': self._help_command,
                '/FAQ': self._faq_command,
                '/define': self._define_command,
                '/learn_words': self._learn_words_command,
                '/vocab_stats': self._vocab_stats_command,
                '/add_words': self._add_words_command,
                '/exit': self._exit_command
            }
            
            if command in command_methods:
                return command_methods[command](args)
            else:
                print("❌ Неизвестная команда. Введите /help для помощи")
                return True
        else:
            # Обычный чат
            response = self.network.chat(user_input, self.current_user_id)
            print(f"🤖 Sin: {response}")
            return True
    
    def _chat_command(self, args):
        if args:
            response = self.network.chat(args, self.current_user_id)
            print(f"🤖 Sin: {response}")
        else:
            print("❌ Укажите сообщение для чата")
        return True
    
    def _learn_command(self, args):
        if args:
            self._learn_text(args, is_new_info=True)
        else:
            print("❌ Укажите текст для обучения")
        return True
    
    def _file_command(self, args):
        if args:
            self._learn_from_file(args)
        else:
            print("❌ Укажите путь к файлу")
        return True
    
    def _url_command(self, args):
        if args:
            self._learn_from_url(args)
        else:
            print("❌ Укажите URL")
        return True
    
    def _auto_command(self, args):
        minutes = 10
        if args:
            try:
                minutes = int(args)
            except:
                print("❌ Неверный формат времени, использую 10 минут")
        self._start_auto_learning(minutes)
        return True
    
    def _stats_command(self, args):
        self._show_stats()
        return True
    
    def _report_command(self, args):
        self._show_comprehensive_report()
        return True
    
    def _save_command(self, args):
        self.network.save_model()
        return True
    
    def _load_command(self, args):
        self.network.load_model()
        return True
    
    def _user_command(self, args):
        if args:
            self.current_user_id = args
            print(f"👥 Пользователь изменен на: {args}")
        else:
            print(f"👥 Текущий пользователь: {self.current_user_id}")
        return True
    
    def _help_command(self, args):
        self.show_welcome()
        return True
    
    def _faq_command(self, args):
        return self.show_faq()
    
    def _define_command(self, args):
        """Команда для поиска значения слова"""
        if args:
            meaning = self.network.vocabulary_system.search_word_meaning(args.strip())
            print(meaning)
        else:
            print("❌ Укажите слово для поиска значения")
        return True
    
    def _learn_words_command(self, args):
        """Команда для автоматического изучения слов"""
        if self.vocabulary_learning_active:
            print("❌ Изучение лексики уже запущено")
            return True
        
        limit = 20  # По умолчанию
        if args:
            try:
                limit = int(args)
                if limit <= 0:
                    limit = 20
            except ValueError:
                print("❌ Неверный формат. Использую значение по умолчанию (20 слов)")
        
        print(f"🚀 Запуск автоматического изучения {limit} слов...")
        
        def learning_worker():
            self.vocabulary_learning_active = True
            try:
                result = self.network.vocabulary_system.auto_learn_vocabulary(limit=limit, delay=2.5)
                print(f"\n✅ Изучение завершено!")
                print(f"   Изучено: {result['learned']} слов")
                print(f"   Ошибок: {result['failed']}")
            except Exception as e:
                logger.error(f"Ошибка в процессе изучения: {e}")
                print(f"❌ Ошибка: {e}")
            finally:
                self.vocabulary_learning_active = False
        
        learning_thread = threading.Thread(target=learning_worker)
        learning_thread.daemon = True
        learning_thread.start()
        
        return True
    
    def _vocab_stats_command(self, args):
        """Команда для показа статистики изучения лексики"""
        stats = self.network.vocabulary_system.get_vocabulary_report()
        print("\n📊 СТАТИСТИКА ИЗУЧЕНИЯ ЛЕКСИКИ:")
        print("-" * 35)
        print(f"Всего изучено слов: {stats['total_learned_words']}")
        print(f"Изучено в этой сессии: {stats['session_learned']}")
        print(f"Ошибок поиска: {stats['failed_lookups']}")
        print(f"Слов в очереди: {stats['queue_size']}")
        print(f"Уникальных слов: {stats['unique_words_studied']}")
        return True
    
    def _add_words_command(self, args):
        """Команда для добавления слов в очередь изучения"""
        if not args:
            print("❌ Укажите слова для добавления (через запятую или пробел)")
            return True
        
        # Разбиваем на слова
        words = re.split(r'[,;\s]+', args)
        words = [word.strip() for word in words if word.strip()]
        
        if not words:
            print("❌ Не найдено корректных слов")
            return True
        
        added_count = self.network.vocabulary_system.add_words_to_learning_queue(words)
        print(f"✅ Добавлено {added_count} слов в очередь изучения")
        
        # Показываем текущий размер очереди
        queue_size = len(self.network.vocabulary_system.learning_queue)
        print(f"📊 Общий размер очереди: {queue_size} слов")
        
        return True
    
    def _exit_command(self, args):
        return False
    
    def _learn_text(self, text, is_new_info=False):
        """Обучение на тексте"""
        print("🧠 Обучение на новом тексте...")
        start_time = time.time()
        
        # Разбиваем на предложения для постепенного обучения
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        useful_info = 0
        total_sentences = len(sentences)
        processed_sentences = 0
        
        for i, sentence in enumerate(sentences[:100]):  # Ограничиваем 100 предложениями
            if sentence and len(sentence) > 5:
                usefulness = self.network.train_on_conversation(
                    [sentence, "Понято"], 
                    is_new_info, 
                    self.current_user_id
                )
                useful_info += usefulness
                processed_sentences += 1
                
                # Показываем прогресс
                if (i + 1) % 20 == 0:
                    progress = (i + 1) / min(100, total_sentences) * 100
                    print(f"📊 Прогресс: {progress:.1f}%")
        
        end_time = time.time()
        avg_usefulness = useful_info / max(1, processed_sentences)
        
        print(f"✅ Обучение завершено!")
        print(f"⏰ Время: {end_time - start_time:.2f} секунд")
        print(f"📈 Полезность: {avg_usefulness:.2f}")
        print(f"📄 Обработано: {processed_sentences} предложений")
    
    def _learn_from_file(self, filepath):
        """Обучение на файле"""
        print(f"📖 Чтение файла: {filepath}")
        
        if not os.path.exists(filepath):
            print("❌ Файл не найден")
            return
        
        # Читаем файл с автоматическим определением кодировки
        text = read_file_with_fallback(filepath)
        
        if text is None:
            print("❌ Ошибка чтения файла - не удалось определить кодировку")
            return
        
        # Определяем тип файла по расширению для дополнительной обработки
        ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if ext == '.docx' and DOCX_AVAILABLE:
                doc = docx.Document(filepath)
                text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            elif ext == '.pdf' and PDF_AVAILABLE:
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = '\n'.join([page.extract_text() for page in pdf_reader.pages])
            elif ext == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Ошибка обработки файла: {e}")
            return
        
        if text:
            print(f"✅ Файл прочитан ({len(text)} символов)")
            self._learn_text(text, is_new_info=True)
        else:
            print("❌ Ошибка чтения файла")
    
    def _learn_from_url(self, url):
        """Обучение по URL"""
        print(f"🌐 Парсинг страницы: {url}")
        start_time = time.time()
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Простая очистка HTML
            clean_text = re.sub(r'<[^>]+>', '', response.text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = clean_text[:20000]  # Ограничиваем размер
            
            if clean_text:
                print(f"✅ Страница загружена ({len(clean_text)} символов)")
                self._learn_text(clean_text, is_new_info=True)
            else:
                print("❌ Пустое содержимое страницы")
        except Exception as e:
            print(f"❌ Ошибка загрузки страницы: {e}")
    
    def _start_auto_learning(self, minutes):
        """Запуск автообучения"""
        if self.auto_learning_active:
            print("❌ Автообучение уже запущено")
            return
        
        print(f"🚀 Запуск автообучения на {minutes} минут...")
        self.auto_learning_active = True
        
        def auto_learning_worker():
            end_time = time.time() + minutes * 60
            generated_texts = []
            iteration = 0
            
            while time.time() < end_time and self.auto_learning_active:
                try:
                    iteration += 1
                    print(f"🔄 Итерация {iteration}...")
                    
                    # Генерируем текст на основе случайных тем
                    topics = ['наука', 'технологии', 'философия', 'искусство', 'история', 'будущее']
                    random_topic = random.choice(topics)
                    
                    prompts = [
                        f"Расскажи о {random_topic}",
                        f"Объясни важность {random_topic}",
                        f"Как {random_topic} влияет на общество?",
                        f"Интересные факты о {random_topic}",
                        f"Мое мнение о {random_topic}"
                    ]
                    
                    prompt = random.choice(prompts)
                    
                    # Генерируем креативный текст
                    generated_text = self.network.creativity_system.generate_creative_response(
                        prompt, 
                        temperature=1.2,
                        max_attempts=2
                    )
                    
                    if len(generated_text) > 15 and self._evaluate_text_quality(generated_text):
                        generated_texts.append(generated_text)
                        
                        # Обучаемся на хорошем тексте
                        self.network.train_on_conversation(
                            [f"Тема: {random_topic}", generated_text],
                            is_new_info=True,
                            user_id=self.current_user_id
                        )
                        
                        print(f"✨ Сохранен хороший текст ({len(generated_text)} символов)")
                        print(f"📝 {generated_text[:100]}...")
                    else:
                        print(f"🗑️ Отброшен низкокачественный текст")
                    
                    # Пауза между генерациями
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Ошибка в автообучении: {e}")
                    time.sleep(5)
            
            self.auto_learning_active = False
            print(f"✅ Автообучение завершено. Сгенерировано {len(generated_texts)} текстов")
        
        learning_thread = threading.Thread(target=auto_learning_worker)
        learning_thread.daemon = True
        learning_thread.start()
    
    def _evaluate_text_quality(self, text):
        """Оценка качества сгенерированного текста"""
        if len(text) < 10:
            return False
        
        # Проверяем на повторяющиеся символы
        if re.search(r'(.)\1{5,}', text):
            return False
        
        # Проверяем соотношение букв и символов
        letters = len(re.findall(r'[а-яА-Яa-zA-Z]', text))
        total_chars = len(text)
        if letters / max(1, total_chars) < 0.4:
            return False
        
        # Проверяем наличие слов
        words = text.split()
        if len(words) < 3:
            return False
        
        return True
    
    def _show_stats(self):
        """Показ статистики"""
        stats = self.network.get_stats()
        print("\n📊 СТАТИСТИКА НЕЙРОСЕТИ SIN:")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Системная статистика
        print(f"\n🖥️ СИСТЕМНАЯ СТАТИСТИКА:")
        print(f"CPU: {psutil.cpu_percent()}%")
        print(f"RAM: {psutil.virtual_memory().percent}%")
        print(f"Время сессии: {int(time.time() - self.session_start_time)} секунд")
    
    def _show_comprehensive_report(self):
        """Показ комплексного отчета"""
        print("\n📋 КОМПЛЕКСНЫЙ ОТЧЕТ О СОСТОЯНИИ SIN:")
        print("=" * 50)
        
        report = self.network.get_comprehensive_report()
        
        # Основные метрики
        print("📈 ОСНОВНЫЕ МЕТРИКИ:")
        stats = report['network_stats']
        print(f"  Нейронов: {stats['total_neurons']}")
        print(f"  Активных: {stats['active_neurons']}")
        print(f"  Синапсов: {stats['synapses']}")
        print(f"  Словарь: {stats['vocab_size']} токенов")
        print(f"  Точность: {stats['performance']:.3f}")
        
        # Эмоциональное состояние
        print("\n😊 ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ:")
        emotions = report['emotional_state']
        print(f"  Настроение: {emotions['emotional_state']}")
        print(f"  Уровень стресса: {emotions['stress_level']:.2f}")
        for emo, value in emotions['detailed_emotions'].items():
            print(f"  {emo.title()}: {value:.2f}")
        
        # Внимание
        print("\n🎯 СИСТЕМА ВНИМАНИЯ:")
        attention = report['attention_state']
        print(f"  Текущий фокус: {attention['current_focus'] or 'Нет'}")
        print(f"  Областей внимания: {attention['total_focus_areas']}")
        if attention['focus_areas']:
            print("  Активные темы:")
            for topic, level in list(attention['focus_areas'].items())[:5]:
                print(f"    - {topic}: {level:.2f}")
        
        # Память
        print("\n🧠 СИСТЕМА ПАМЯТИ:")
        memory = report['memory_state']
        print(f"  Долговременная: {memory['long_term_memory_size']}")
        print(f"  Краткосрочная: {memory['short_term_memory_size']}")
        print(f"  Забыто: {memory['forgotten_count']}")
        
        # Креативность
        print("\n🎨 СИСТЕМА КРЕАТИВНОСТИ:")
        creativity = report['creativity_state']
        print(f"  Уровень креативности: {creativity['overall_creativity_level']:.2f}")
        print(f"  Средняя свежесть: {creativity['average_recent_creativity']:.2f}")
        
        # Социальное взаимодействие
        print("\n👥 СОЦИАЛЬНАЯ СИСТЕМА:")
        social = report['social_state']
        print(f"  Пользователей: {social['total_users']}")
        print(f"  Взаимодействий: {social['total_interactions']}")
        print(f"  Средний уровень отношений: {social['average_relationship_score']:.2f}")
        
        # Изучение лексики
        print("\n📚 СИСТЕМА ИЗУЧЕНИЯ ЛЕКСИКИ:")
        vocab = report['vocabulary_learning_state']
        print(f"  Всего изучено слов: {vocab['total_learned_words']}")
        print(f"  Изучено в сессии: {vocab['session_learned']}")
        print(f"  Слов в очереди: {vocab['queue_size']}")
        print(f"  Уникальных слов: {vocab['unique_words_studied']}")
    
    def run(self):
        """Основной цикл работы"""
        self.show_welcome()
        
        try:
            while True:
                try:
                    user_input = input(f"\n👤 {self.current_user_id}: ").strip()
                    if not user_input:
                        continue
                    
                    if not self.process_command(user_input):
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n👋 До свидания! Спасибо за общение!")
                    break
                except Exception as e:
                    logger.error(f"❌ Ошибка: {e}")
                    print("❌ Произошла ошибка. Попробуйте еще раз.")
        
        finally:
            # Сохраняем модель при выходе
            print("\n💾 Сохранение модели...")
            self.network.save_model()
            print("✅ Готово!")

# Создание необходимых файлов и директорий при запуске
def initialize_sin_environment():
    """Инициализация рабочей среды для Sin"""
    # Создаем директории
    directories = ['models', 'data', 'logs', 'datasets', 'sessions']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Создаем пустые файлы если их нет
    files_to_create = [
        'sin_model.pkl',
        'sin_tokenizer_vocab.pkl',
        'sin_ai.log'
    ]
    
    for file_path in files_to_create:
        if not os.path.exists(file_path):
            try:
                with open(file_path, 'w') as f:
                    pass  # Создаем пустой файл
            except Exception as e:
                logger.warning(f"Не удалось создать файл {file_path}: {e}")

if __name__ == "__main__":
    # Инициализируем среду
    initialize_sin_environment()
    
    # Запускаем консольный интерфейс Sin
    console = SinConsole()
    console.run()
