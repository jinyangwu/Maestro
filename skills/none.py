

class NoSkill:
    def __init__(self, skill, client, model, system_prompt, question, images, temperature, top_p, timeout, have_image=False):
        self.skill = skill
        self.client = client
        self.model = model
        # self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.have_image = have_image

        if images == []:
            messages=[{"role": "user", "content": question}]
        else:
            messages_text=[{"role": "user", "content": question}]
            # 构建包含图片的消息内容
            if system_prompt == "":
                messages = []
            else:
                messages = [{"role": "system", "content": system_prompt}]
            content = []

            # content = [{"type": "text", "text": "Describe this picture."}]
            
            # 添加所有图片
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"{image}"
                    }
                })
            content.append({"type": "text", "text": question})
            
            messages.append({"role": "user", "content": content})
        self.messages = messages

    def generate(self):
        try:
            outputs = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout
            )
            return outputs
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return ""

    
