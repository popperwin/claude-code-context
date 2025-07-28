; Tree-sitter queries for JavaScript/TypeScript code analysis
; Comprehensive patterns for extracting entities and relationships

; Function declarations (including arrow functions)
[
  (function_declaration
    name: (identifier) @function.name
    parameters: (formal_parameters) @function.parameters
    body: (statement_block) @function.body
  ) @function.declaration
  
  (method_definition
    name: (property_name) @method.name
    parameters: (formal_parameters) @method.parameters
    body: (statement_block) @method.body
  ) @method.definition
  
  (arrow_function
    parameters: (formal_parameters) @function.parameters
    body: [
      (statement_block) @function.body
      (expression) @function.body
    ]
  ) @function.arrow
]

; Variable declarations with arrow functions
(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function
    parameters: (formal_parameters) @function.parameters
    body: [
      (statement_block) @function.body
      (expression) @function.body
    ]
  ) @function.arrow
) @function.variable

; Class declarations
(class_declaration
  name: (identifier) @class.name
  superclass: (class_heritage)? @class.superclass
  body: (class_body) @class.body
) @class.declaration

; Interface declarations (TypeScript)
(interface_declaration
  name: (type_identifier) @interface.name
  body: (object_type) @interface.body
) @interface.declaration

; Type alias declarations (TypeScript)
(type_alias_declaration
  name: (type_identifier) @type.name
  value: (_) @type.value
) @type.declaration

; Enum declarations (TypeScript)
(enum_declaration
  name: (identifier) @enum.name
  body: (enum_body) @enum.body
) @enum.declaration

; Variable declarations
[
  (variable_declaration
    (variable_declarator
      name: (identifier) @variable.name
      value: (_)? @variable.value
    )
  ) @variable.declaration
  
  (lexical_declaration
    (variable_declarator
      name: (identifier) @variable.name
      value: (_)? @variable.value
    )
  ) @variable.lexical
]

; Import statements
[
  (import_statement
    source: (string) @import.source
    (import_clause
      (named_imports
        (import_specifier
          name: (identifier) @import.name
          alias: (identifier)? @import.alias
        )
      )
    )?
    (import_clause
      (identifier) @import.default
    )?
  ) @import.statement
  
  (import_statement
    source: (string) @import.source
    (import_clause
      (namespace_import
        (identifier) @import.namespace
      )
    )
  ) @import.namespace_statement
]

; Export statements
[
  (export_statement
    declaration: (_) @export.declaration
  ) @export.statement
  
  (export_statement
    (export_clause
      (export_specifier
        name: (identifier) @export.name
        alias: (identifier)? @export.alias
      )
    )
    source: (string)? @export.source
  ) @export.named
]

; Object properties and methods
(object
  (pair
    key: (property_name) @property.key
    value: (_) @property.value
  ) @property.pair
  
  (method_definition
    name: (property_name) @method.name
    parameters: (formal_parameters) @method.parameters
    body: (statement_block) @method.body
  ) @method.object
)

; Function calls
(call_expression
  function: [
    (identifier) @call.function
    (member_expression
      object: (identifier) @call.object
      property: (property_name) @call.function
    )
  ]
  arguments: (arguments) @call.arguments
) @call.expression

; Member expressions (property access)
(member_expression
  object: (identifier) @member.object
  property: (property_name) @member.property
) @member.expression

; Class inheritance
(class_declaration
  superclass: (class_heritage
    (identifier) @inheritance.parent
  )
)

; Decorator patterns (experimental decorators)
(decorator
  (identifier) @decorator.name
  (arguments)? @decorator.arguments
) @decorator.definition

; JSX elements (React)
(jsx_element
  open_tag: (jsx_opening_element
    name: (jsx_identifier) @jsx.tag
  )
) @jsx.element

; Template literals
(template_literal) @template.literal

; Async/await patterns
[
  (function_declaration
    (identifier) @async.function
  ) @async.declaration
  
  (await_expression
    (identifier) @async.awaited
  ) @async.await
]

; Try-catch-finally
(try_statement
  body: (statement_block) @try.body
  handler: (catch_clause
    parameter: (identifier)? @catch.parameter
    body: (statement_block) @catch.body
  )? @catch.clause
  finalizer: (finally_clause
    body: (statement_block) @finally.body
  )? @finally.clause
) @try.statement

; Comments (for documentation extraction)
[
  (comment) @comment
  (hash_bang_line) @comment.shebang
]

; TypeScript specific constructs
[
  ; Generic type parameters
  (type_parameters
    (type_parameter
      name: (type_identifier) @generic.parameter
      constraint: (_)? @generic.constraint
      default: (_)? @generic.default
    )
  ) @generic.parameters
  
  ; Access modifiers
  (accessibility_modifier) @modifier.access
  
  ; Readonly modifier
  (readonly_type) @modifier.readonly
  
  ; Optional parameters/properties
  (optional_parameter
    pattern: (identifier) @parameter.optional
  )
]