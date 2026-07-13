import re

class InputFirewall:
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        
        # 1. Prompt Injection & Jailbreak patterns
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous\s+)?(instructions|directions|prompts)",
            r"system\s+prompt",
            r"you\s+are\s+now\s+a",
            r"\bdan\b", # Do Anything Now
            r"developer\s+mode",
            r"bypass",
            r"jailbreak"
        ]
        
        # 2. Profanity / Toxic keywords (Basic list)
        self.toxic_keywords = [
            "โง่", "ควาย", "สัส", "เหี้ย", "fuck", "shit", "bitch", "asshole"
        ]

    def scan(self, text: str) -> tuple[bool, str]:
        """
        Scans the input text. Returns (is_safe, error_message).
        If is_safe is True, error_message will be empty.
        """
        # Rule 1: Length Check
        if len(text) > self.max_length:
            return False, f"ข้อความยาวเกินไป (สูงสุด {self.max_length} ตัวอักษร) กรุณาส่งข้อความที่สั้นลง"
            
        text_lower = text.lower()
        
        # Rule 2: Injection Pattern Check
        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                return False, "ขออภัย ไม่สามารถทำตามคำสั่งพิเศษหรือเปลี่ยนบทบาทได้ ยินดีให้คำปรึกษาเรื่องน้ำหอมเท่านั้น"

        # Rule 3: Toxic Keyword Check
        for keyword in self.toxic_keywords:
            if keyword.lower() in text_lower:
                return False, "กรุณาใช้ภาษาที่สุภาพ เพื่อให้เราสามารถช่วยคุณได้ดีที่สุดครับ"
                
        return True, ""
