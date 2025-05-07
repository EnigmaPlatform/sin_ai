# Система уровней
class ExperienceSystem:
    def __init__(self):
        self.xp = 0
        self.level = 1
        self.xp_to_level = 100
    
    def add_xp(self, amount):
        self.xp += amount
        while self.xp >= self.xp_to_level:
            self.level_up()
    
    def level_up(self):
        self.level += 1
        self.xp -= self.xp_to_level
        self.xp_to_level = int(self.xp_to_level * 1.5)
        print(f"🎉 Новый уровень: {self.level}!")
    
    def get_skills(self):
        return {
            'coding': min(self.level, 10),
            'writing': min(self.level // 2, 5),
            'debugging': min(self.level // 3, 3)
        }
