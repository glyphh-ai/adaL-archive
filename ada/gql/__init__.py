"""
Glyph Query Language (GQL) - A declarative query language for Ada's hyperdimensional semantic space.

GQL provides SQL-like syntax for querying, comparing, and analyzing glyphs. It supports:
- Similarity search across the glyph space
- List and count operations with filtering
- Comparison between glyphs at different hierarchy levels
- Temporal prediction and drift detection
- Introspection and trend analysis
- Aggregation with grouping

Example:
    >>> from ada.gql import parse, execute
    >>> 
    >>> # Parse and execute a GQL query
    >>> query = 'FIND SIMILAR TO "red sports car" LIMIT 10 THRESHOLD 0.7'
    >>> ast = parse(query)
    >>> result = execute(ast, model)
    >>> 
    >>> # Result is always a FactTree
    >>> print(result.to_text())

The GQL system consists of:
- Lexer: Tokenizes GQL source into tokens
- Parser: Builds AST from tokens
- Planner: Transforms AST into execution plans
- Executor: Executes plans against SDK operations
- Translator: Converts natural language to GQL
"""

from ada.gql.exceptions import (
    GQLError,
    LexerError,
    ParseError,
    PlanningError,
    ExecutionError,
    SlotValidationError,
    HierarchyError,
    PredictionError,
)

from ada.gql.tokens import Token, TokenType, KEYWORDS
from ada.gql.lexer import GQLLexer
from ada.gql.ast import (
    ASTNode,
    ASTVisitor,
    HierarchyLevel,
    GlyphRef,
    Condition,
    ComparisonCondition,
    LogicalCondition,
    SimilaritySearchNode,
    ListNode,
    CountNode,
    CompareNode,
    PredictNode,
    DriftDetectionNode,
    IntrospectNode,
    TrendNode,
    AggregateNode,
    ast_from_dict,
    condition_from_dict,
)
from ada.gql.parser import GQLParser, parse
from ada.gql.printer import GQLPrettyPrinter, pretty_print
from ada.gql.encoder import QueryEncoder
from ada.gql.cache import QueryCache, CacheEntry, CacheStats
from ada.gql.plans import (
    ExecutionPlan,
    SimilaritySearchPlan,
    ListPlan,
    CountPlan,
    ComparePlan,
    PredictPlan,
    DriftPlan,
    IntrospectPlan,
    TrendPlan,
    AggregatePlan,
)
from ada.gql.planner import ExecutionPlanner, ExecutionContext
from ada.gql.executor import GQLExecutor, execute
from ada.gql.storage import GlyphStorageProtocol, InMemoryGlyphStorage
from ada.gql.patterns import (
    SlotType,
    SlotDefinition,
    GQLPattern,
    DEFAULT_GQL_PATTERNS,
)
from ada.gql.translator import NLTranslator, TranslationResult
from ada.gql.stored_procedure import StoredProcedure

__all__ = [
    # Exceptions
    "GQLError",
    "LexerError",
    "ParseError",
    "PlanningError",
    "ExecutionError",
    "SlotValidationError",
    "HierarchyError",
    "PredictionError",
    # Tokens
    "Token",
    "TokenType",
    "KEYWORDS",
    # Lexer
    "GQLLexer",
    # AST
    "ASTNode",
    "ASTVisitor",
    "HierarchyLevel",
    "GlyphRef",
    "Condition",
    "ComparisonCondition",
    "LogicalCondition",
    "SimilaritySearchNode",
    "ListNode",
    "CountNode",
    "CompareNode",
    "PredictNode",
    "DriftDetectionNode",
    "IntrospectNode",
    "TrendNode",
    "AggregateNode",
    "ast_from_dict",
    "condition_from_dict",
    # Parser
    "GQLParser",
    "parse",
    # Pretty Printer
    "GQLPrettyPrinter",
    "pretty_print",
    # Query Encoder
    "QueryEncoder",
    # Query Cache
    "QueryCache",
    "CacheEntry",
    "CacheStats",
    # Execution Plans
    "ExecutionPlan",
    "SimilaritySearchPlan",
    "ListPlan",
    "CountPlan",
    "ComparePlan",
    "PredictPlan",
    "DriftPlan",
    "IntrospectPlan",
    "TrendPlan",
    "AggregatePlan",
    # Planner
    "ExecutionPlanner",
    "ExecutionContext",
    # Executor
    "GQLExecutor",
    "execute",
    # Storage
    "GlyphStorageProtocol",
    "InMemoryGlyphStorage",
    # Patterns
    "SlotType",
    "SlotDefinition",
    "GQLPattern",
    "DEFAULT_GQL_PATTERNS",
    # Translator
    "NLTranslator",
    "TranslationResult",
    # Stored Procedures
    "StoredProcedure",
]
