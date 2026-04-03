// Basic String Methods
s.length()
// length of string
s.chatAt(i)
// char at index i
s.substring(a, b)
// substring from a to b
s.substring(a)
// substring from a to end
s.equalsIgnoreCase(t)
// ignore case of t
s.compareTo(t)
// lexicographical order

  // Case Conversion
Character.isLowerCase(c)
Character.isUpperCase(c)
Character.toLowerCase(c)
Character.toUpperCase(c)

// Reverse String
new StringBuilder(s).reverse().toString()

// Checking Palindrome
s.equals(new StringBuilder(s).reverse().toString())

// Splitting 
s.split(" ")
s.split("\\s+")

// Trim Spaces 
s.trim()

// Character Type Checks
Character.isLetter(c)
Character.isDigit(c)
Character.isLetterOrDigit(c)
