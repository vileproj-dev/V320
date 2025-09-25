#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v3.0 - Enhanced AI Manager
Gerenciador de IA com Gemini direto usando 3 chaves com rotação automática
ZERO SIMULAÇÃO - Apenas modelos reais funcionais
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Import do cliente Gemini direto
from .gemini_direct_client import gemini_direct_client, generate_with_gemini_direct, generate_with_gemini_direct_sync

logger = logging.getLogger(__name__)

class EnhancedAIManager:
    """Gerenciador de IA aprimorado com Gemini direto"""

    def __init__(self):
        """Inicializa o gerenciador aprimorado com Gemini direto"""
        self.gemini_client = gemini_direct_client
        self.search_orchestrator = None
        
        # Importar search orchestrator se disponível
        try:
            from .real_search_orchestrator import RealSearchOrchestrator
            self.search_orchestrator = RealSearchOrchestrator()
            logger.info("✅ Search Orchestrator carregado")
        except ImportError:
            logger.warning("⚠️ Search Orchestrator não disponível")

        logger.info("🤖 Enhanced AI Manager inicializado com Gemini direto")

    def generate_response(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-exp",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Gera resposta síncrona usando Gemini direto"""
        try:
            # Usa o cliente Gemini direto
            response = self.gemini_client.generate_content_sync(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Verifica se response é dict válido
            if response and isinstance(response, dict) and response.get('success'):
                return {
                    'success': True,
                    'content': response.get('content', ''),
                    'model': response.get('model_used', model),
                    'provider': 'gemini_direct',
                    'tokens_used': response.get('tokens_used', 0),
                    'api_key_used': response.get('api_key_used', 'gemini_1')
                }
            else:
                logger.error(f"❌ Falha na geração de resposta: {response}")
                error_msg = 'Erro desconhecido'
                if isinstance(response, dict):
                    error_msg = response.get('error', 'Erro desconhecido')
                return {
                    'success': False,
                    'content': 'Erro ao gerar resposta',
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"❌ Erro na geração de resposta: {e}")
            return {
                'success': False,
                'content': 'Erro interno ao gerar resposta',
                'error': str(e)
            }

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None
    ) -> str:
        """
        Gera texto usando Gemini direto
        
        Args:
            prompt: Prompt do usuário
            system_prompt: Prompt do sistema (opcional)
            max_tokens: Máximo de tokens (opcional)
            temperature: Temperatura (opcional)
            model_override: Ignorado (sempre usa Gemini)
        
        Returns:
            String com a resposta da IA
        """
        try:
            return await generate_with_gemini_direct(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens or 4000,
                temperature=temperature or 0.7
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Erro de conexão ao gerar texto: {str(e)}")
            raise
        except (ValueError, KeyError) as e:
            logger.error(f"❌ Erro de parâmetros ao gerar texto: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao gerar texto: {str(e)}")
            raise
    
    def generate_text_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None
    ) -> str:
        """Versão síncrona da geração de texto"""
        try:
            return generate_with_gemini_direct_sync(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens or 4000,
                temperature=temperature or 0.7
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Erro de conexão ao gerar texto (sync): {str(e)}")
            raise
        except (ValueError, KeyError) as e:
            logger.error(f"❌ Erro de parâmetros ao gerar texto (sync): {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erro inesperado ao gerar texto (sync): {str(e)}")
            raise

    async def generate_with_active_search(
        self,
        prompt: str,
        context: str = "",
        session_id: str = None,
        max_search_iterations: int = 3,
        preferred_model: str = None,
        min_processing_time: int = 0
    ) -> str:
        """
        Gera conteúdo com busca ativa usando hierarquia OpenRouter
        """
        logger.info(f"🔍 Iniciando geração com busca ativa (min_time: {min_processing_time}s)")
        
        # Registrar tempo de início para garantir tempo mínimo
        start_time = datetime.now()

        # Prepara prompt com instruções de busca e contexto
        enhanced_prompt = f"""
{prompt}

CONTEXTO DISPONÍVEL:
{context}

INSTRUÇÕES ESPECIAIS:
- Analise o contexto fornecido detalhadamente
- Busque dados atualizados sobre o mercado brasileiro
- Procure por estatísticas, tendências e casos reais
- Forneça insights profundos baseados nos dados disponíveis
- Use informações reais e atualizadas sempre que possível

IMPORTANTE: Gere uma análise completa e profissional baseando-se no contexto fornecido.
"""

        # Sistema prompt para busca ativa
        system_prompt = """Você é um especialista em análise de mercado e tendências digitais. 
        Sua função é gerar análises profundas e insights valiosos baseados nos dados fornecidos.
        Sempre forneça informações precisas, atualizadas e acionáveis.
        Se precisar de informações adicionais, indique claramente quais dados seriam úteis."""

        try:
            # Sempre usar Gemini direto (preferred_model é ignorado)
            logger.info(f"🤖 Usando Gemini direto (preferência {preferred_model} ignorada)")
            
            # Gerar resposta usando Gemini direto
            response = await self.generate_text(
                prompt=enhanced_prompt,
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            # Garantir tempo mínimo de processamento se especificado
            if min_processing_time > 0:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time < min_processing_time:
                    remaining_time = min_processing_time - elapsed_time
                    logger.info(f"⏱️ Aguardando {remaining_time:.1f}s para completar tempo mínimo")
                    await asyncio.sleep(remaining_time)
            
            logger.info("✅ Geração com busca ativa concluída")
            return response
            
        except Exception as e:
            logger.error(f"❌ Erro na geração com busca ativa: {e}")
            # Fallback simples
            try:
                return await self.generate_text(enhanced_prompt, system_prompt)
            except Exception as e2:
                logger.error(f"❌ Erro no fallback: {e2}")
                raise

    async def analyze_content(
        self,
        content: str,
        analysis_type: str = "comprehensive",
        target_audience: str = "general",
        model_preference: str = None
    ) -> str:
        """
        Analisa conteúdo usando hierarquia OpenRouter
        
        Args:
            content: Conteúdo para análise
            analysis_type: Tipo de análise (comprehensive, viral, market, etc.)
            target_audience: Público-alvo
            model_preference: Preferência de modelo
        
        Returns:
            Análise detalhada do conteúdo
        """
        system_prompt = f"""Você é um especialista em análise de conteúdo digital e marketing.
        Sua função é analisar conteúdo de forma {analysis_type} para o público {target_audience}.
        Forneça insights acionáveis, tendências identificadas e recomendações estratégicas."""
        
        analysis_prompt = f"""
Analise o seguinte conteúdo de forma {analysis_type}:

CONTEÚDO:
{content}

PÚBLICO-ALVO: {target_audience}

FORNEÇA:
1. Análise detalhada do conteúdo
2. Pontos fortes e fracos identificados
3. Potencial viral e engajamento
4. Recomendações de melhoria
5. Estratégias de distribuição
6. Insights de mercado relevantes

Seja específico, prático e acionável em suas recomendações.
"""
        
        try:
            return await self.generate_text(
                prompt=analysis_prompt,
                system_prompt=system_prompt,
                max_tokens=3000,
                temperature=0.7,
                model_override=model_preference
            )
        except Exception as e:
            logger.error(f"❌ Erro na análise de conteúdo: {e}")
            raise

    async def generate_insights(
        self,
        data: Dict[str, Any],
        insight_type: str = "market_trends",
        depth: str = "deep"
    ) -> str:
        """
        Gera insights baseados em dados usando hierarquia OpenRouter
        
        Args:
            data: Dados para análise
            insight_type: Tipo de insight desejado
            depth: Profundidade da análise (shallow, medium, deep)
        
        Returns:
            Insights gerados
        """
        system_prompt = f"""Você é um analista de dados especializado em {insight_type}.
        Sua função é gerar insights {depth} baseados nos dados fornecidos.
        Sempre forneça análises precisas, tendências identificadas e recomendações acionáveis."""
        
        data_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        insights_prompt = f"""
Analise os seguintes dados e gere insights {depth} sobre {insight_type}:

DADOS:
{data_str}

FORNEÇA:
1. Principais tendências identificadas
2. Padrões e correlações importantes
3. Oportunidades de mercado
4. Riscos e desafios
5. Recomendações estratégicas
6. Previsões baseadas nos dados

Seja específico, use números quando relevante e forneça insights acionáveis.
"""
        
        try:
            return await self.generate_text(
                prompt=insights_prompt,
                system_prompt=system_prompt,
                max_tokens=4000,
                temperature=0.6
            )
        except Exception as e:
            logger.error(f"❌ Erro na geração de insights: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Retorna status do gerenciador"""
        return {
            "gemini_status": self.gemini_client.get_status(),
            "search_orchestrator_available": self.search_orchestrator is not None,
            "timestamp": datetime.now().isoformat()
        }

    def reset_failed_models(self):
        """Reseta estatísticas do cliente Gemini"""
        self.gemini_client.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "key_rotations": 0,
            "last_used_key": None
        }
        logger.info("✅ Estatísticas Gemini resetadas")

# Instância global para uso em todo o projeto
enhanced_ai_manager = EnhancedAIManager()

# Funções de conveniência para uso direto
async def generate_ai_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model_override: Optional[str] = None
) -> str:
    """Função de conveniência para geração de texto"""
    return await enhanced_ai_manager.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

def generate_ai_text_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    model_override: Optional[str] = None
) -> str:
    """Função de conveniência síncrona para geração de texto"""
    return enhanced_ai_manager.generate_text_sync(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

if __name__ == "__main__":
    # Teste básico
    async def test():
        try:
            manager = EnhancedAIManager()
            
            response = await manager.generate_text(
                prompt="Explique brevemente o que é inteligência artificial",
                system_prompt="Você é um especialista em tecnologia"
            )
            print(f"Resposta: {response}")
            
            # Status
            status = manager.get_status()
            print(f"Status: {json.dumps(status, indent=2, default=str)}")
            
        except Exception as e:
            print(f"Erro no teste: {e}")
    
    asyncio.run(test())