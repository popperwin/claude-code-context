; Tree-sitter queries for Python code analysis
; Comprehensive patterns for extracting entities and relationships

; Function definitions (including async)
[
  (function_definition
    name: (identifier) @function.name
    parameters: (parameters) @function.parameters
    return_type: (type)? @function.return_type
    body: (block) @function.body
  ) @function.definition
  
  (decorated_definition
    (decorator)+ @function.decorators
    definition: (function_definition
      name: (identifier) @function.name
      parameters: (parameters) @function.parameters
      return_type: (type)? @function.return_type
      body: (block) @function.body
    ) @function.definition
  ) @function.decorated
]

; Class definitions
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.superclasses
  body: (block) @class.body
) @class.definition

; Method definitions (functions inside classes)
(class_definition
  body: (block
    [
      (function_definition
        name: (identifier) @method.name
        parameters: (parameters) @method.parameters
        return_type: (type)? @method.return_type
        body: (block) @method.body
      ) @method.definition
      
      (decorated_definition
        (decorator)+ @method.decorators
        definition: (function_definition
          name: (identifier) @method.name
          parameters: (parameters) @method.parameters
          return_type: (type)? @method.return_type
          body: (block) @method.body
        ) @method.definition
      ) @method.decorated
    ]
  )
)

; Import statements
[
  (import_statement
    name: (dotted_name) @import.name
  ) @import.statement
  
  (import_from_statement
    module_name: (dotted_name) @import.module
    name: [
      (dotted_name) @import.name
      (aliased_import 
        name: (dotted_name) @import.name
        alias: (identifier) @import.alias
      )
      (import_list
        (dotted_name) @import.name
      )
      (import_list
        (aliased_import
          name: (dotted_name) @import.name
          alias: (identifier) @import.alias
        )
      )
    ]
  ) @import.from_statement
]

; Variable assignments
(assignment
  left: [
    (identifier) @variable.name
    (pattern_list (identifier) @variable.name)
    (attribute (identifier) @variable.object attribute: (identifier) @variable.name)
  ]
  type: (type)? @variable.type
  right: (_) @variable.value
) @variable.assignment

; Type annotations
(variable_declaration
  (identifier) @variable.name
  type: (type) @variable.type
  value: (_)? @variable.value
) @variable.declaration

; Function calls
(call
  function: [
    (identifier) @call.function
    (attribute 
      object: (identifier) @call.object
      attribute: (identifier) @call.function
    )
  ]
  arguments: (argument_list) @call.arguments
) @call.expression

; Decorators
(decorator
  (identifier) @decorator.name
  (argument_list)? @decorator.arguments
) @decorator.definition

; Class inheritance
(class_definition
  superclasses: (argument_list
    [
      (identifier) @inheritance.parent
      (attribute
        object: (identifier) @inheritance.parent_module
        attribute: (identifier) @inheritance.parent
      )
    ]
  )
)

; String literals (for docstrings)
(expression_statement
  (string) @docstring
) @docstring.statement

; Constants (uppercase variables)
(assignment
  left: (identifier) @constant.name
  right: (_) @constant.value
) @constant.assignment

; Exception handling
[
  (try_statement) @exception.try
  (except_clause
    type: (identifier)? @exception.type
  ) @exception.except
  (raise_statement
    (identifier)? @exception.raised
  ) @exception.raise
]

; Async/await patterns
[
  (function_definition
    body: (block
      (expression_statement
        (await
          argument: (_) @async.awaited
        )
      )
    )
  )
  
  (await
    argument: (_) @async.awaited
  ) @async.await
]

; Property definitions
(decorated_definition
  (decorator
    (identifier) @property.decorator
  )
  definition: (function_definition
    name: (identifier) @property.name
  )
) @property.definition