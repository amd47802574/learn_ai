import os
import sys
import json
import time
from typing import List, Optional
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class EmbeddingTester:
    def __init__(self):
        """初始化Embedding测试器"""
        self.api_key = os.getenv('EMBEDDING_API_KEY')
        self.base_url = os.getenv('EMBEDDING_BASE_URL')
        self.model = os.getenv('EMBEDDING_MODEL')
        
        # 验证配置
        if not self.api_key or self.api_key == 'xxxx':
            raise ValueError("❌ EMBEDDING_API_KEY 未配置或使用默认值，请检查 .env 文件")
        
        if not self.base_url:
            raise ValueError("❌ EMBEDDING_BASE_URL 未配置，请检查 .env 文件")
        
        if not self.model:
            raise ValueError("❌ EMBEDDING_MODEL 未配置，请检查 .env 文件")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        print(f"✅ 配置信息:")
        print(f"   - Base URL: {self.base_url}")
        print(f"   - Model: {self.model}")
        print(f"   - API Key: {self.api_key[:8]}...{self.api_key[-4:] if len(self.api_key) > 12 else '***'}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取单个文本的embedding向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 获取embedding失败: {str(e)}")
            return None
    
    def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量获取文本的embedding向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ 批量获取embedding失败: {str(e)}")
            return None
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        cosine_sim = np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
        return float(cosine_sim)
    
    def test_basic(self):
        """基础功能测试"""
        print("\n" + "="*60)
        print("🔍 开始基础功能测试")
        print("="*60)
        
        # 测试文本
        test_texts = [
            "今天天气很好",
            "人工智能技术发展迅速",
            "Python是一种编程语言"
        ]
        
        # 测试单个embedding
        print("\n📝 测试1: 获取单个文本embedding")
        embedding = self.get_embedding(test_texts[0])
        if embedding:
            print(f"   ✅ 成功获取embedding，维度: {len(embedding)}")
            print(f"   前10个值: {embedding[:10]}")
        else:
            print("   ❌ 获取embedding失败")
            return False
        
        # 测试批量embedding
        print("\n📝 测试2: 批量获取embedding")
        embeddings = self.get_embeddings(test_texts)
        if embeddings:
            print(f"   ✅ 成功获取{len(embeddings)}个embedding")
            for i, emb in enumerate(embeddings):
                print(f"      - 文本{i+1}维度: {len(emb)}")
        else:
            print("   ❌ 批量获取embedding失败")
            return False
        
        return True
    
    def test_similarity(self):
        """语义相似度测试"""
        print("\n" + "="*60)
        print("🔍 开始语义相似度测试")
        print("="*60)
        
        # 测试用例：相似文本对
        test_cases = [
            {
                "text1": "人工智能是未来发展趋势",
                "text2": "AI将成为未来主流技术",
                "expected": "高相似度"
            },
            {
                "text1": "我喜欢吃苹果",
                "text2": "今天天气不错",
                "expected": "低相似度"
            },
            {
                "text1": "机器学习需要大量数据",
                "text2": "深度学习依赖海量样本",
                "expected": "中等相似度"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 测试用例 {i}:")
            print(f"   文本1: {case['text1']}")
            print(f"   文本2: {case['text2']}")
            print(f"   预期: {case['expected']}")
            
            embeddings = self.get_embeddings([case['text1'], case['text2']])
            if embeddings and len(embeddings) == 2:
                similarity = self.calculate_similarity(embeddings[0], embeddings[1])
                print(f"   ✅ 余弦相似度: {similarity:.4f}")
                
                # 简单判断合理性
                if case['expected'] == "高相似度" and similarity < 0.7:
                    print(f"   ⚠️  相似度低于预期，可能存在问题")
                elif case['expected'] == "低相似度" and similarity > 0.3:
                    print(f"   ⚠️  相似度高于预期，可能存在问题")
                else:
                    print(f"   ✅ 相似度结果合理")
            else:
                print(f"   ❌ 获取embedding失败")
                return False
        
        return True
    
    def test_performance(self):
        """性能测试"""
        print("\n" + "="*60)
        print("🔍 开始性能测试")
        print("="*60)
        
        # 不同长度的文本
        test_texts = [
            "短文本",
            "这是一个中等长度的文本用于测试embedding性能",
            "这是一个较长的文本。它包含了更多的内容和信息。" * 10,
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 测试文本 {i} (长度: {len(text)} 字符):")
            
            start_time = time.time()
            embedding = self.get_embedding(text)
            elapsed_time = time.time() - start_time
            
            if embedding:
                print(f"   ✅ 耗时: {elapsed_time:.3f} 秒")
                print(f"   向量维度: {len(embedding)}")
                if elapsed_time > 2.0:
                    print(f"   ⚠️  响应时间较长: {elapsed_time:.3f}秒")
            else:
                print(f"   ❌ 获取embedding失败")
                return False
        
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("🚀 开始Embedding服务全面测试")
        print("="*60)
        
        tests = [
            ("基础功能测试", self.test_basic),
            ("语义相似度测试", self.test_similarity),
            ("性能测试", self.test_performance)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n❌ 测试 '{test_name}' 异常: {str(e)}")
                failed += 1
        
        # 总结
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)
        print(f"✅ 通过: {passed} 项")
        print(f"❌ 失败: {failed} 项")
        
        if failed == 0:
            print("\n🎉 所有测试通过！Embedding服务正常工作！")
            return True
        else:
            print("\n⚠️  部分测试失败，请检查服务配置或网络连接")
            return False

def main():
    """主函数"""
    try:
        tester = EmbeddingTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
        print("\n请确保 .env 文件配置正确:")
        print("  EMBEDDING_API_KEY=你的API密钥")
        print("  EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1")
        print("  EMBEDDING_MODEL=text-embedding-v4")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()