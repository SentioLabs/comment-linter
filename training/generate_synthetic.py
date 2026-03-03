"""Generate synthetic labeled training data for the comment superfluousness classifier.

Produces realistic feature vectors and comment text for both superfluous and
valuable comments across all 5 supported languages. Uses domain knowledge about
what makes comments superfluous vs valuable to create diverse, balanced data.

Usage:
    python generate_synthetic.py --output synthetic.jsonl --count 8000
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Comment templates — realistic text patterns per category
# ---------------------------------------------------------------------------

SUPERFLUOUS_TEMPLATES = {
    "restate_code": [
        # Directly mirror what the code does
        "increment counter",
        "increment the counter",
        "increment i",
        "decrement the counter",
        "get user name",
        "get the user name",
        "get username",
        "set user name",
        "set the value",
        "set config value",
        "return the result",
        "return result",
        "return the value",
        "return true",
        "return false",
        "return nil",
        "return None",
        "return error",
        "check if null",
        "check if nil",
        "check for nil",
        "check if empty",
        "check if valid",
        "validate input",
        "validate input data",
        "validate the input",
        "validate request",
        "validate params",
        "parse the input",
        "parse input string",
        "parse the response",
        "parse JSON",
        "parse json response",
        "convert to string",
        "convert to int",
        "convert to integer",
        "convert to float",
        "cast to string",
        "create new instance",
        "create a new user",
        "create new connection",
        "create new client",
        "initialize the database",
        "initialize database connection",
        "initialize the logger",
        "initialize config",
        "init the server",
        "open the file",
        "open file for reading",
        "close the file",
        "close the connection",
        "close database connection",
        "delete the item",
        "delete user",
        "delete the record",
        "remove the element",
        "remove item from list",
        "add to list",
        "add item to the list",
        "add element to array",
        "append to array",
        "push to stack",
        "pop from stack",
        "send the request",
        "send HTTP request",
        "send the message",
        "send email",
        "read the file",
        "read config file",
        "read from database",
        "write to file",
        "write the output",
        "write to database",
        "update the record",
        "update user data",
        "update the cache",
        "save the result",
        "save to database",
        "save user data",
        "load the config",
        "load configuration",
        "load user data",
        "fetch the data",
        "fetch user data",
        "fetch from API",
        "call the API",
        "call the function",
        "call the method",
        "handle the error",
        "handle error",
        "handle the exception",
        "catch the error",
        "log the error",
        "log the message",
        "print the result",
        "print output",
        "format the string",
        "format output",
        "format the date",
        "sort the list",
        "sort the array",
        "sort by name",
        "filter the results",
        "filter by status",
        "filter items",
        "map over items",
        "calculate total",
        "calculate total price",
        "calculate the sum",
        "compute the hash",
        "compute checksum",
        "encode the data",
        "encode to base64",
        "decode the response",
        "decode from base64",
        "encrypt the password",
        "hash the password",
        "compare passwords",
        "check password",
        "authenticate user",
        "authorize the request",
        "start the server",
        "start the timer",
        "stop the timer",
        "stop the server",
        "reset the counter",
        "reset state",
        "clear the cache",
        "clear the list",
        "flush the buffer",
        "connect to database",
        "connect to server",
        "disconnect from server",
        "clone the object",
        "copy the data",
        "merge the results",
        "split the string",
        "join the strings",
        "trim whitespace",
        "normalize the path",
        "resolve the path",
        "find the element",
        "find user by id",
        "search for item",
        "lookup the value",
        "get or create",
        "find or create user",
        "execute the query",
        "execute SQL query",
        "run the command",
        "run the migration",
        "process the request",
        "process the data",
        "process items",
        "transform the data",
        "serialize to JSON",
        "deserialize from JSON",
        "marshal the struct",
        "unmarshal the response",
    ],
    "section_label": [
        "Variables",
        "Constants",
        "Imports",
        "Types",
        "Interfaces",
        "Structs",
        "Classes",
        "Constructor",
        "Destructor",
        "Getters",
        "Setters",
        "Getters and setters",
        "Public methods",
        "Private methods",
        "Helper methods",
        "Utility functions",
        "Static methods",
        "Event handlers",
        "Lifecycle methods",
        "Callbacks",
        "Middleware",
        "Routes",
        "Endpoints",
        "Configuration",
        "Setup",
        "Teardown",
        "Initialization",
        "Cleanup",
        "Error handling",
        "Logging",
        "Validation",
        "Authentication",
        "Authorization",
        "Database",
        "Cache",
        "State management",
        "Rendering",
        "Main logic",
        "Business logic",
        "Core logic",
        "Helpers",
        "Utils",
        "Tests",
        "Mocks",
        "Fixtures",
    ],
    "obvious_loop": [
        "loop through items",
        "loop through the list",
        "iterate over items",
        "iterate over the array",
        "iterate through results",
        "for each item",
        "for each user",
        "for each element",
        "loop over all entries",
        "traverse the tree",
        "walk the directory",
        "iterate over keys",
        "iterate over values",
        "loop through map",
        "process each item",
        "handle each element",
    ],
    "obvious_conditional": [
        "check if exists",
        "check if true",
        "check if false",
        "if error return",
        "if nil return",
        "if null return",
        "if empty return",
        "guard clause",
        "early return",
        "check condition",
        "boundary check",
        "null check",
        "nil check",
        "empty check",
        "type check",
        "error check",
    ],
    "trivial_assignment": [
        "assign the value",
        "set default value",
        "set to zero",
        "set to empty",
        "set to nil",
        "set to null",
        "set to false",
        "initialize to zero",
        "initialize to empty",
        "default value",
        "default config",
        "assign result",
        "store the result",
        "save the value",
        "update the field",
    ],
}

VALUABLE_TEMPLATES = {
    "why_explanation": [
        "We use a mutex here because the upstream API is not thread-safe",
        "Using exponential backoff because the service rate-limits aggressively",
        "This is intentionally sequential to avoid overwhelming the database",
        "We clone here because the borrow checker cannot prove the reference outlives the closure",
        "Manual implementation needed because serde cannot derive for this enum variant",
        "Retry logic added after production incident PROJ-1234 revealed transient failures",
        "Sorting before dedup because the input may contain non-adjacent duplicates",
        "Using a channel instead of shared state to avoid deadlock risk in the pipeline",
        "Explicit type annotation required because the compiler cannot infer through the trait object",
        "Pre-allocating capacity because the profiler showed reallocation as a bottleneck",
        "Casting to i64 because the API returns timestamps that overflow i32 after 2038",
        "Buffered writer used because unbuffered I/O was causing 10x slowdown on large files",
        "Using raw SQL instead of ORM because the query requires window functions",
        "Manual JSON parsing because the response format is inconsistent across API versions",
        "Locking order matters here: always acquire user lock before account lock to prevent deadlock",
        "We suppress this lint because the unsafe block is required for FFI compatibility",
        "Using a weak reference to break a reference cycle between parent and child nodes",
        "Spawning on a blocking thread because this calls synchronous filesystem APIs",
        "Connection pooling disabled for tests because it masks connection leak bugs",
        "Zero-copy parsing here saves 200MB RSS on the production dataset",
        "The custom comparator handles NaN values which sort unstably with default ordering",
        "We normalize Unicode to NFC because the database collation is case-insensitive",
        "Using a BTreeMap instead of HashMap because we need deterministic iteration order for snapshots",
        "Flushing before fork because child processes inherit unflushed buffers",
        "Using wrapping arithmetic because overflow is expected in the hash function",
        "Two-phase commit needed because the operation spans multiple data stores",
        "Pinning the future because it contains a self-referential struct",
        "This timeout is intentionally generous because some clients are on satellite links",
        "Rate limit is per-tenant, not global, to prevent noisy-neighbor issues",
        "Padding the struct to a cache line to prevent false sharing in concurrent access",
    ],
    "external_reference": [
        "See https://github.com/golang/go/issues/12345 for the upstream bug",
        "See RFC 3339 for the expected timestamp format",
        "See https://docs.rs/tokio/latest/tokio/sync for async primitives",
        "Per OWASP recommendation A03:2021 — Injection Prevention",
        "Follows the pattern from https://martinfowler.com/bliki/CircuitBreaker.html",
        "See CVE-2022-12345 for why we sanitize this input",
        "Reference implementation at https://github.com/example/lib/blob/main/src/core.rs",
        "Spec: https://www.w3.org/TR/wai-aria-practices/#dialog_modal",
        "Based on algorithm from Cormen et al., Introduction to Algorithms, Ch. 13",
        "See PostgreSQL docs on advisory locks: https://www.postgresql.org/docs/current/explicit-locking.html",
        "Per PCI DSS requirement 3.4: render PAN unreadable anywhere it is stored",
        "Based on the retry strategy described in AWS Architecture Blog post on exponential backoff",
        "See https://developer.mozilla.org/en-US/docs/Web/API/AbortController for cancellation",
        "Implements the Raft consensus algorithm (https://raft.github.io/raft.pdf)",
        "See POSIX spec for signal handling: https://pubs.opengroup.org/onlinepubs/9699919799/",
    ],
    "todo_with_ticket": [
        "TODO: Remove after migration to v3 completes (PROJ-456)",
        "TODO: Replace with native fetch once we drop IE11 support (PROJ-789)",
        "TODO(team): Switch to the new auth provider after Q4 rollout (JIRA-123)",
        "FIXME: This workaround can be removed once upstream fixes issue #567",
        "HACK: Temporary until the schema migration in PROJ-890 lands",
        "TODO: Replace manual retry with circuit breaker pattern (PROJ-234)",
        "TODO: Remove this compatibility shim after all clients upgrade to v2 (PROJ-345)",
        "FIXME: Race condition here — tracked in PROJ-678, needs proper locking",
        "TODO: Extract into a shared library once the API stabilizes (PROJ-901)",
        "TODO: Add rate limiting before public launch (SEC-012)",
    ],
    "workaround": [
        "Workaround for upstream bug in net/http where connections leak under high concurrency",
        "Workaround for a known memory leak in the WebSocket library when connections are not properly closed",
        "Workaround for a race condition in the event loop when multiple subscribers attach simultaneously",
        "Workaround for serde bug #4321 where flattened enums lose their tag",
        "Workaround: the ORM generates invalid SQL for self-joins, so we use raw query",
        "Temporary workaround for timezone handling bug in chrono < 0.5",
        "Workaround for Docker DNS resolution failures during rolling deploys",
        "Workaround for gRPC keepalive not working through AWS NLB",
    ],
    "caveat": [
        "Note: this only strips ASCII control characters; Unicode normalization is handled separately",
        "Caveat: the cache is not invalidated on config changes; restart required",
        "Warning: changing this value requires a database migration",
        "Important: this function is not reentrant; callers must hold the lock",
        "Note: the order of these cases matters; most specific must come first",
        "Caveat: this assumes UTC timestamps; local time will produce incorrect results",
        "Warning: exceeding this limit will trigger the circuit breaker for 30 seconds",
        "Note: this allocation is intentional; the profiler confirmed it is cheaper than recomputing",
        "Important: do not reorder these fields; the binary protocol depends on layout order",
        "Caveat: this timeout must be longer than the database statement timeout to avoid double-retry",
    ],
    "legacy_compat": [
        "Legacy compatibility: the old serialization format used little-endian byte order",
        "Legacy: older clients send timestamps as i32 epoch seconds instead of i64",
        "Legacy compatibility: the v1 API used XML responses; this adapter converts to JSON",
        "Backward compat: the old SDK passed auth tokens in query params instead of headers",
        "Legacy: maintaining this codepath until all tenants migrate to the new billing system",
        "Backward compatibility: the old format used comma-separated values instead of JSON arrays",
        "Legacy: this field name was renamed in v2 but we still accept the old name",
        "Legacy compatibility shim: remove after all clients upgrade past v1.2.0",
    ],
    "design_decision": [
        "Using an arena allocator to keep all AST nodes in contiguous memory for cache locality",
        "Deliberately not using generics here to keep the API surface simple for plugin authors",
        "Chose eventual consistency over strong consistency for better availability under partition",
        "Using optimistic locking because contention is low and retries are cheap",
        "Streaming response to avoid buffering the entire dataset in memory",
        "Using a trie instead of a hash map because we need prefix matching",
        "Accepting the N+1 query here because batching would complicate the transaction boundary",
        "Using HTTP long-polling instead of WebSockets for better compatibility with corporate proxies",
        "Keeping this as a monolith until we have clear service boundaries",
        "Using compile-time dispatch via generics rather than dynamic dispatch for the hot path",
    ],
}

LANGUAGES = ["go", "rust", "python", "typescript", "javascript"]

FILE_EXTENSIONS = {
    "go": ".go",
    "rust": ".rs",
    "python": ".py",
    "typescript": ".ts",
    "javascript": ".js",
}

ADJACENT_NODE_KINDS = {
    "go": [
        "function_declaration", "method_declaration", "short_var_declaration",
        "assignment_statement", "if_statement", "for_statement", "return_statement",
        "call_expression", "var_declaration", "const_declaration", "type_declaration",
    ],
    "rust": [
        "function_item", "let_declaration", "expression_statement", "if_expression",
        "for_expression", "return_expression", "call_expression", "struct_item",
        "impl_item", "use_declaration", "const_item", "match_expression",
    ],
    "python": [
        "function_definition", "assignment", "if_statement", "for_statement",
        "return_statement", "call", "class_definition", "import_statement",
        "expression_statement", "with_statement", "try_statement",
    ],
    "typescript": [
        "function_declaration", "variable_declaration", "if_statement",
        "for_statement", "return_statement", "call_expression",
        "class_declaration", "import_statement", "export_statement",
        "arrow_function", "method_definition", "interface_declaration",
    ],
    "javascript": [
        "function_declaration", "variable_declaration", "if_statement",
        "for_statement", "return_statement", "call_expression",
        "class_declaration", "import_statement", "export_statement",
        "arrow_function", "method_definition", "assignment_expression",
    ],
}


def generate_superfluous_features(rng: random.Random, category: str) -> dict:
    """Generate realistic feature vectors for superfluous comments."""
    base = {
        "token_overlap_jaccard": 0.0,
        "identifier_substring_ratio": 0.0,
        "comment_token_count": 0,
        "is_doc_comment": False,
        "is_before_declaration": False,
        "is_inline": False,
        "nesting_depth": 0,
        "has_why_indicator": False,
        "has_external_ref": False,
        "imperative_verb_noun": False,
        "is_section_label": False,
        "contains_literal_values": False,
        "references_other_files": False,
        "references_specific_functions": False,
        "mirrors_data_structure": False,
        "comment_code_age_ratio": None,
    }

    if category == "restate_code":
        base["token_overlap_jaccard"] = rng.uniform(0.3, 0.95)
        base["identifier_substring_ratio"] = rng.uniform(0.2, 0.9)
        base["comment_token_count"] = rng.randint(2, 6)
        base["imperative_verb_noun"] = rng.random() < 0.7
        base["is_before_declaration"] = rng.random() < 0.6
        base["is_inline"] = rng.random() < 0.3
        base["nesting_depth"] = rng.choice([0, 0, 1, 1, 2])

    elif category == "section_label":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.3)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.2)
        base["comment_token_count"] = rng.randint(1, 3)
        base["is_section_label"] = True
        base["imperative_verb_noun"] = rng.random() < 0.1
        base["is_before_declaration"] = rng.random() < 0.4
        base["nesting_depth"] = rng.choice([0, 0, 0, 1])

    elif category == "obvious_loop":
        base["token_overlap_jaccard"] = rng.uniform(0.15, 0.6)
        base["identifier_substring_ratio"] = rng.uniform(0.1, 0.5)
        base["comment_token_count"] = rng.randint(3, 5)
        base["imperative_verb_noun"] = rng.random() < 0.6
        base["is_before_declaration"] = rng.random() < 0.3
        base["is_inline"] = rng.random() < 0.2
        base["nesting_depth"] = rng.choice([1, 1, 2, 2, 3])

    elif category == "obvious_conditional":
        base["token_overlap_jaccard"] = rng.uniform(0.1, 0.5)
        base["identifier_substring_ratio"] = rng.uniform(0.1, 0.4)
        base["comment_token_count"] = rng.randint(2, 5)
        base["imperative_verb_noun"] = rng.random() < 0.3
        base["is_before_declaration"] = rng.random() < 0.2
        base["is_inline"] = rng.random() < 0.4
        base["nesting_depth"] = rng.choice([1, 1, 2, 2, 3])

    elif category == "trivial_assignment":
        base["token_overlap_jaccard"] = rng.uniform(0.2, 0.7)
        base["identifier_substring_ratio"] = rng.uniform(0.2, 0.8)
        base["comment_token_count"] = rng.randint(2, 5)
        base["imperative_verb_noun"] = rng.random() < 0.5
        base["is_before_declaration"] = rng.random() < 0.5
        base["is_inline"] = rng.random() < 0.3
        base["nesting_depth"] = rng.choice([0, 1, 1, 2])

    # Occasional noise: some superfluous comments happen to have these
    if rng.random() < 0.02:
        base["contains_literal_values"] = True
    if rng.random() < 0.05:
        base["is_doc_comment"] = True
    if rng.random() < 0.03:
        base["comment_code_age_ratio"] = round(rng.uniform(0.8, 1.2), 2)

    return base


def generate_valuable_features(rng: random.Random, category: str) -> dict:
    """Generate realistic feature vectors for valuable comments."""
    base = {
        "token_overlap_jaccard": 0.0,
        "identifier_substring_ratio": 0.0,
        "comment_token_count": 0,
        "is_doc_comment": False,
        "is_before_declaration": False,
        "is_inline": False,
        "nesting_depth": 0,
        "has_why_indicator": False,
        "has_external_ref": False,
        "imperative_verb_noun": False,
        "is_section_label": False,
        "contains_literal_values": False,
        "references_other_files": False,
        "references_specific_functions": False,
        "mirrors_data_structure": False,
        "comment_code_age_ratio": None,
    }

    if category == "why_explanation":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.15)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.1)
        base["comment_token_count"] = rng.randint(8, 25)
        base["has_why_indicator"] = True
        base["is_before_declaration"] = rng.random() < 0.4
        base["is_inline"] = rng.random() < 0.3
        base["nesting_depth"] = rng.choice([0, 1, 1, 2, 2, 3])

    elif category == "external_reference":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.1)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.05)
        base["comment_token_count"] = rng.randint(6, 20)
        base["has_external_ref"] = True
        base["has_why_indicator"] = rng.random() < 0.3
        base["is_before_declaration"] = rng.random() < 0.3
        base["nesting_depth"] = rng.choice([0, 0, 1, 1, 2])

    elif category == "todo_with_ticket":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.1)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.1)
        base["comment_token_count"] = rng.randint(6, 15)
        base["has_external_ref"] = True
        base["has_why_indicator"] = rng.random() < 0.4
        base["is_before_declaration"] = rng.random() < 0.2
        base["nesting_depth"] = rng.choice([0, 1, 1, 2])
        base["references_specific_functions"] = rng.random() < 0.2

    elif category == "workaround":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.15)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.1)
        base["comment_token_count"] = rng.randint(8, 20)
        base["has_why_indicator"] = True
        base["has_external_ref"] = rng.random() < 0.5
        base["is_before_declaration"] = rng.random() < 0.5
        base["nesting_depth"] = rng.choice([0, 1, 1, 2])
        base["references_specific_functions"] = rng.random() < 0.3

    elif category == "caveat":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.15)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.1)
        base["comment_token_count"] = rng.randint(8, 20)
        base["has_why_indicator"] = rng.random() < 0.6
        base["is_before_declaration"] = rng.random() < 0.3
        base["is_inline"] = rng.random() < 0.2
        base["nesting_depth"] = rng.choice([0, 1, 1, 2])
        base["contains_literal_values"] = rng.random() < 0.3

    elif category == "legacy_compat":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.15)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.1)
        base["comment_token_count"] = rng.randint(8, 18)
        base["has_why_indicator"] = rng.random() < 0.5
        base["has_external_ref"] = rng.random() < 0.3
        base["is_before_declaration"] = rng.random() < 0.4
        base["nesting_depth"] = rng.choice([0, 0, 1, 1])
        base["references_specific_functions"] = rng.random() < 0.2

    elif category == "design_decision":
        base["token_overlap_jaccard"] = rng.uniform(0.0, 0.2)
        base["identifier_substring_ratio"] = rng.uniform(0.0, 0.15)
        base["comment_token_count"] = rng.randint(10, 25)
        base["has_why_indicator"] = True
        base["is_before_declaration"] = rng.random() < 0.6
        base["nesting_depth"] = rng.choice([0, 0, 1])
        base["references_specific_functions"] = rng.random() < 0.3

    # Valuable comments sometimes have doc-comment style
    if rng.random() < 0.15:
        base["is_doc_comment"] = True
    # Sometimes reference other files or functions
    if rng.random() < 0.1:
        base["references_other_files"] = True
    # Age ratio: valuable comments tend to be older (stable)
    if rng.random() < 0.2:
        base["comment_code_age_ratio"] = round(rng.uniform(0.5, 2.0), 2)

    return base


def make_record(
    rng: random.Random,
    label: int,
    comment_text: str,
    features: dict,
    language: str,
    label_source: str = "synthetic",
) -> dict:
    """Assemble a complete JSONL record."""
    ext = FILE_EXTENSIONS[language]
    node_kinds = ADJACENT_NODE_KINDS[language]
    adj = rng.choice(node_kinds)
    features["adjacent_node_kind"] = adj

    # Synthesize a plausible file path
    dirs = ["src", "internal", "pkg", "lib", "core", "api", "cmd", "util"]
    files = ["main", "handler", "service", "model", "config", "cache",
             "auth", "db", "router", "middleware", "utils", "helpers",
             "client", "server", "worker", "pipeline", "parser"]
    path = f"{rng.choice(dirs)}/{rng.choice(files)}{ext}"

    comment_kind = "line"
    if features.get("is_doc_comment"):
        comment_kind = "doc"
    elif rng.random() < 0.1:
        comment_kind = "block"

    return {
        "file": path,
        "line": rng.randint(1, 500),
        "column": rng.choice([0, 0, 0, 4, 4, 8, 8, 12]),
        "language": language,
        "comment_text": comment_text,
        "comment_kind": comment_kind,
        "heuristic_score": 0.0,  # not used for training, placeholder
        "heuristic_confidence": 0.0,
        "features": features,
        "label": label,
        "label_source": label_source,
    }


def generate_dataset(
    count: int = 8000,
    seed: int = 42,
    superfluous_ratio: float = 0.40,
) -> list[dict]:
    """Generate a balanced synthetic dataset.

    Args:
        count: Total number of records to generate.
        seed: Random seed.
        superfluous_ratio: Fraction of superfluous examples (rest are valuable).

    Returns:
        List of JSONL records.
    """
    rng = random.Random(seed)
    records = []

    n_superfluous = int(count * superfluous_ratio)
    n_valuable = count - n_superfluous

    sup_categories = list(SUPERFLUOUS_TEMPLATES.keys())
    val_categories = list(VALUABLE_TEMPLATES.keys())

    # Generate superfluous examples
    for _ in range(n_superfluous):
        cat = rng.choice(sup_categories)
        text = rng.choice(SUPERFLUOUS_TEMPLATES[cat])
        lang = rng.choice(LANGUAGES)
        features = generate_superfluous_features(rng, cat)
        records.append(make_record(rng, 1, text, features, lang))

    # Generate valuable examples
    for _ in range(n_valuable):
        cat = rng.choice(val_categories)
        text = rng.choice(VALUABLE_TEMPLATES[cat])
        lang = rng.choice(LANGUAGES)
        features = generate_valuable_features(rng, cat)
        records.append(make_record(rng, 0, text, features, lang))

    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for comment classifier"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--count", type=int, default=8000,
        help="Total number of examples (default: 8000)",
    )
    parser.add_argument(
        "--superfluous-ratio", type=float, default=0.40,
        help="Fraction of superfluous examples (default: 0.40)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    records = generate_dataset(
        count=args.count,
        seed=args.seed,
        superfluous_ratio=args.superfluous_ratio,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Report
    labels = [r["label"] for r in records]
    n_sup = sum(1 for l in labels if l == 1)
    n_val = sum(1 for l in labels if l == 0)
    print(f"Generated {len(records)} synthetic examples:", file=sys.stderr)
    print(f"  Superfluous: {n_sup} ({n_sup/len(records)*100:.1f}%)", file=sys.stderr)
    print(f"  Valuable:    {n_val} ({n_val/len(records)*100:.1f}%)", file=sys.stderr)

    # Category distribution
    from collections import Counter
    sup_cats = Counter()
    val_cats = Counter()
    for r in records:
        text = r["comment_text"]
        if r["label"] == 1:
            for cat, templates in SUPERFLUOUS_TEMPLATES.items():
                if text in templates:
                    sup_cats[cat] += 1
                    break
        else:
            for cat, templates in VALUABLE_TEMPLATES.items():
                if text in templates:
                    val_cats[cat] += 1
                    break

    print("\n  Superfluous categories:", file=sys.stderr)
    for cat, cnt in sup_cats.most_common():
        print(f"    {cat}: {cnt}", file=sys.stderr)
    print("  Valuable categories:", file=sys.stderr)
    for cat, cnt in val_cats.most_common():
        print(f"    {cat}: {cnt}", file=sys.stderr)


if __name__ == "__main__":
    main()
