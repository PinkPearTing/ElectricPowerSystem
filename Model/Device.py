class Devices:
    def __init__(self, insulators=None, arrestors=None, transformers=None):
        self.insulators = insulators or []
        self.arrestors = arrestors or []
        self.transformers = transformers or []

    def add_insolator(self, insolator):
        """
        添加绝缘子
        """
        self.insulators.append(insolator)

    def add_arrestor(self, arrestor):
        """
        添加避雷器
        """
        self.arrestors.append(arrestor)

    def add_transformer(self, transformer):
        """
        添加变压器
        """
        self.transformers.append(transformer)
    
    def __repr__(self):
        """
        返回对象的字符串表示形式。
        """
        return f"Device(INS={self.insulators}, SAR={self.arrestors}, TXF={self.transformers})"