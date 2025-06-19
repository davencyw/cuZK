#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <functional>
#include "../poseidon/field_arithmetic.hpp"

namespace MVPolynomial {

using FieldElement = Poseidon::FieldElement;

// Represents a monomial term with variables and their degrees
// For example, x_0^2 * x_1 * x_3^3 would be represented as {0: 2, 1: 1, 3: 3}
class Monomial {
public:
    std::map<size_t, size_t> variables; // variable_index -> degree
    
    Monomial() = default;
    explicit Monomial(const std::map<size_t, size_t>& vars);
    
    // Comparison for ordering in maps
    bool operator==(const Monomial& other) const;
    bool operator<(const Monomial& other) const;
    
    // Monomial operations
    Monomial operator*(const Monomial& other) const;
    
    // Get total degree
    size_t total_degree() const;
    
    // Get the maximum variable index
    size_t max_variable_index() const;
    
    // Evaluate monomial at given point
    FieldElement evaluate(const std::vector<FieldElement>& point) const;
    
    // Convert to string for debugging
    std::string to_string() const;
};

// Hash function for Monomial to use in unordered_map
struct MonomialHash {
    size_t operator()(const Monomial& m) const;
};

// A multivariate polynomial represented as a sum of terms
// Each term is a coefficient multiplied by a monomial
class MultivariatePolynomial {
private:
    std::unordered_map<Monomial, FieldElement, MonomialHash> terms;
    size_t num_variables;
    
public:
    // Constructors
    MultivariatePolynomial();
    explicit MultivariatePolynomial(size_t num_vars);
    MultivariatePolynomial(const std::unordered_map<Monomial, FieldElement, MonomialHash>& terms, size_t num_vars);
    
    // Static constructors for common cases
    static MultivariatePolynomial zero(size_t num_vars);
    static MultivariatePolynomial one(size_t num_vars);
    static MultivariatePolynomial variable(size_t var_index, size_t num_vars);
    static MultivariatePolynomial constant(const FieldElement& coeff, size_t num_vars);
    
    // Coefficient access
    void set_coefficient(const Monomial& monomial, const FieldElement& coeff);
    FieldElement get_coefficient(const Monomial& monomial) const;
    void add_term(const Monomial& monomial, const FieldElement& coeff);
    
    // Basic properties
    size_t get_num_variables() const { return num_variables; }
    size_t total_degree() const;
    bool is_zero() const;
    size_t num_terms() const { return terms.size(); }
    
    // Polynomial operations
    MultivariatePolynomial operator+(const MultivariatePolynomial& other) const;
    MultivariatePolynomial operator-(const MultivariatePolynomial& other) const;
    MultivariatePolynomial operator*(const MultivariatePolynomial& other) const;
    MultivariatePolynomial operator*(const FieldElement& scalar) const;
    
    MultivariatePolynomial& operator+=(const MultivariatePolynomial& other);
    MultivariatePolynomial& operator-=(const MultivariatePolynomial& other);
    MultivariatePolynomial& operator*=(const MultivariatePolynomial& other);
    MultivariatePolynomial& operator*=(const FieldElement& scalar);
    
    // Evaluation
    FieldElement evaluate(const std::vector<FieldElement>& point) const;
    
    // Partial evaluation - fix some variables to specific values
    MultivariatePolynomial partial_evaluate(const std::map<size_t, FieldElement>& assignments) const;
    
    // For sumcheck: evaluate with the last variable being the free variable
    // Returns a univariate polynomial in the last variable
    std::vector<FieldElement> evaluate_partial_to_univariate(
        const std::vector<FieldElement>& fixed_values, size_t max_degree = 0) const;
    
    // Utility functions
    std::string to_string() const;
    void cleanup(); // Remove zero coefficients
    
    // Iterator access to terms
    const std::unordered_map<Monomial, FieldElement, MonomialHash>& get_terms() const { return terms; }
};

// Utility functions for common polynomial constructions
namespace Utils {
    // Create a random polynomial of given degree
    MultivariatePolynomial random_polynomial(size_t num_vars, size_t max_degree);
    
    // Create a polynomial that represents the Boolean hypercube constraint
    // Returns a polynomial that is 0 when each variable is 0 or 1
    MultivariatePolynomial boolean_constraint(size_t num_vars);
    
    // Create a multilinear extension of a function over the Boolean hypercube
    MultivariatePolynomial multilinear_extension(
        const std::vector<FieldElement>& values, size_t num_vars);
}

} // namespace MVPolynomial 