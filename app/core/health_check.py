"""
Health Check and Monitoring System
Provides comprehensive health checks for all system components
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging
import httpx

from .config import settings


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "service": self.service_name,
            "status": self.status.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class HealthChecker:
    """Base class for health checkers"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    async def check(self) -> HealthCheckResult:
        """
        Perform health check
        Must be implemented by subclasses
        """
        raise NotImplementedError
    
    def _measure_time(self, func):
        """Measure execution time of a function"""
        start = time.time()
        try:
            result = func()
            elapsed = (time.time() - start) * 1000
            return result, elapsed
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            raise


class OllamaHealthChecker(HealthChecker):
    """Health checker for Ollama service"""
    
    def __init__(self):
        super().__init__("Ollama LLM")
        self.base_url = settings.OLLAMA_BASE_URL
    
    async def check(self) -> HealthCheckResult:
        """Check Ollama service health"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    return HealthCheckResult(
                        service_name=self.service_name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=(time.time() - start_time) * 1000,
                        message=f"Ollama API returned status {response.status_code}",
                        details={"url": self.base_url}
                    )
                
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                # Check if required model is available
                required_model = settings.OLLAMA_MODEL
                model_available = any(
                    required_model in name for name in model_names
                )
                
                if not model_available:
                    return HealthCheckResult(
                        service_name=self.service_name,
                        status=HealthStatus.DEGRADED,
                        response_time_ms=(time.time() - start_time) * 1000,
                        message=f"Model '{required_model}' not found",
                        details={
                            "available_models": model_names,
                            "required_model": required_model
                        }
                    )
                
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="Ollama is running with required model",
                    details={
                        "model": required_model,
                        "available_models_count": len(models)
                    }
                )
                
        except httpx.ConnectError:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message="Cannot connect to Ollama service",
                details={
                    "url": self.base_url,
                    "suggestion": "Make sure Ollama is running"
                }
            )
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database"""
    
    def __init__(self):
        super().__init__("Database")
    
    async def check(self) -> HealthCheckResult:
        """Check database health"""
        start_time = time.time()
        
        try:
            # Import here to avoid circular dependencies
            from pathlib import Path
            import sqlite3
            
            db_path = settings.DATABASE_URL.replace("sqlite:///", "")
            
            # Check if database file exists
            if not Path(db_path).exists():
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.DEGRADED,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="Database file does not exist yet",
                    details={
                        "path": db_path,
                        "suggestion": "Database will be created on first use"
                    }
                )
            
            # Try to connect and execute a simple query
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            # Get database size
            db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message="Database is accessible",
                details={
                    "path": db_path,
                    "size_mb": round(db_size_mb, 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Database health check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )


class ChromaDBHealthChecker(HealthChecker):
    """Health checker for ChromaDB vector store"""
    
    def __init__(self):
        super().__init__("ChromaDB")
    
    async def check(self) -> HealthCheckResult:
        """Check ChromaDB health"""
        start_time = time.time()
        
        try:
            from pathlib import Path
            import chromadb
            
            # Check if persist directory exists
            persist_dir = Path(settings.CHROMA_PERSIST_DIRECTORY)
            if not persist_dir.exists():
                return HealthCheckResult(
                    service_name=self.service_name,
                    status=HealthStatus.DEGRADED,
                    response_time_ms=(time.time() - start_time) * 1000,
                    message="ChromaDB directory does not exist yet",
                    details={
                        "path": str(persist_dir),
                        "suggestion": "Will be created when first document is processed"
                    }
                )
            
            # Try to connect to ChromaDB
            client = chromadb.PersistentClient(path=str(persist_dir))
            
            # Try to get or create collection
            collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME
            )
            
            # Get collection count
            count = collection.count()
            
            # Calculate storage size
            storage_size_mb = sum(
                f.stat().st_size for f in persist_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024)
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message="ChromaDB is operational",
                details={
                    "collection": settings.CHROMA_COLLECTION_NAME,
                    "document_count": count,
                    "storage_mb": round(storage_size_mb, 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"ChromaDB health check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )


class EmbeddingModelHealthChecker(HealthChecker):
    """Health checker for embedding model"""
    
    def __init__(self):
        super().__init__("Embedding Model")
    
    async def check(self) -> HealthCheckResult:
        """Check embedding model health"""
        start_time = time.time()
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check if model can be loaded
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            # Test embedding generation
            test_text = "Health check test"
            embedding = model.encode([test_text])
            
            # Check device
            device = "cuda" if torch.cuda.is_available() and settings.EMBEDDING_DEVICE == "cuda" else "cpu"
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message="Embedding model loaded successfully",
                details={
                    "model": settings.EMBEDDING_MODEL,
                    "device": device,
                    "embedding_dim": len(embedding[0])
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Embedding model health check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )


class DiskSpaceHealthChecker(HealthChecker):
    """Health checker for disk space"""
    
    def __init__(self):
        super().__init__("Disk Space")
    
    async def check(self) -> HealthCheckResult:
        """Check available disk space"""
        start_time = time.time()
        
        try:
            import shutil
            from pathlib import Path
            
            # Check disk space for data directory
            data_path = Path("./data")
            total, used, free = shutil.disk_usage(data_path)
            
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            used_percent = (used / total) * 100
            
            # Determine status based on free space
            if free_gb < 1.0:
                status = HealthStatus.UNHEALTHY
                message = "Critical: Less than 1GB free space"
            elif free_gb < 5.0:
                status = HealthStatus.DEGRADED
                message = "Warning: Less than 5GB free space"
            else:
                status = HealthStatus.HEALTHY
                message = "Sufficient disk space available"
            
            return HealthCheckResult(
                service_name=self.service_name,
                status=status,
                response_time_ms=(time.time() - start_time) * 1000,
                message=message,
                details={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=self.service_name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Disk space check failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )


class SystemHealthMonitor:
    """
    Comprehensive system health monitor
    Coordinates all health checks and provides aggregated status
    """
    
    def __init__(self):
        self.checkers: List[HealthChecker] = [
            OllamaHealthChecker(),
            DatabaseHealthChecker(),
            ChromaDBHealthChecker(),
            EmbeddingModelHealthChecker(),
            DiskSpaceHealthChecker()
        ]
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks
        
        Returns:
            Dictionary with overall status and individual check results
        """
        logger.info("🏥 Running system health checks...")
        
        # Run all checks concurrently
        results = await asyncio.gather(
            *[checker.check() for checker in self.checkers],
            return_exceptions=True
        )
        
        # Process results
        check_results = []
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed with exception: {result}")
                check_results.append(
                    HealthCheckResult(
                        service_name="Unknown",
                        status=HealthStatus.UNKNOWN,
                        response_time_ms=0,
                        message=str(result)
                    )
                )
                status_counts[HealthStatus.UNKNOWN] += 1
            else:
                check_results.append(result)
                status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        response = {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [result.to_dict() for result in check_results],
            "summary": {
                "total_checks": len(check_results),
                "healthy": status_counts[HealthStatus.HEALTHY],
                "degraded": status_counts[HealthStatus.DEGRADED],
                "unhealthy": status_counts[HealthStatus.UNHEALTHY],
                "unknown": status_counts[HealthStatus.UNKNOWN]
            }
        }
        
        logger.info(
            f"Health check complete | "
            f"Overall: {overall_status.value} | "
            f"Healthy: {status_counts[HealthStatus.HEALTHY]}/{len(check_results)}"
        )
        
        return response


# Global health monitor instance
health_monitor = SystemHealthMonitor()


async def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return await health_monitor.check_all()


if __name__ == "__main__":
    # Test health checks
    async def test():
        health = await get_system_health()
        
        import json
        print(json.dumps(health, indent=2))
    
    asyncio.run(test())
