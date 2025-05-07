# Анализ Python-кода
import ast
import radon
from radon.complexity import cc_visit

class CodeAnalyzer:
    def analyze(self, code):
        try:
            tree = ast.parse(code)
            metrics = {
                'functions': [],
                'complexity': self._calculate_complexity(code),
                'style_issues': self._check_style(code)
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['functions'].append({
                        'name': node.name,
                        'args': len(node.args.args),
                        'lines': len(node.body)
                    })
            
            return metrics
        except SyntaxError as e:
            return {'error': str(e)}

    def _calculate_complexity(self, code):
        results = cc_visit(code)
        return sum(m.complexity for m in results)

    def _check_style(self, code):
        issues = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 79:
                issues.append(f"Line {i}: Too long ({len(line)} chars)")
            if '\t' in line:
                issues.append(f"Line {i}: Contains tabs")
        return issues
