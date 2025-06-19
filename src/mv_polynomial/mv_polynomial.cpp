#include "mv_polynomial.hpp"
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>

namespace MVPolynomial {

// Monomial implementation
Monomial::Monomial(const std::map<size_t, size_t>& vars) : variables(vars) {
    // Remove zero degrees
    auto it = variables.begin();
    while (it != variables.end()) {
        if (it->second == 0) {
            it = variables.erase(it);
        } else {
            ++it;
        }
    }
}

bool Monomial::operator==(const Monomial& other) const {
    return variables == other.variables;
}

bool Monomial::operator<(const Monomial& other) const {
    return variables < other.variables;
}

Monomial Monomial::operator*(const Monomial& other) const {
    std::map<size_t, size_t> result_vars = variables;
    
    for (const auto& [var, degree] : other.variables) {
        result_vars[var] += degree;
    }
    
    return Monomial(result_vars);
}

size_t Monomial::total_degree() const {
    size_t total = 0;
    for (const auto& [var, degree] : variables) {
        total += degree;
    }
    return total;
}

size_t Monomial::max_variable_index() const {
    if (variables.empty()) return 0;
    return variables.rbegin()->first;
}

FieldElement Monomial::evaluate(const std::vector<FieldElement>& point) const {
    using namespace Poseidon;
    
    FieldElement result = FieldConstants::ONE;
    
    for (const auto& [var_index, degree] : variables) {
        if (var_index >= point.size()) {
            throw std::invalid_argument("Variable index out of bounds");
        }
        
        // Compute point[var_index]^degree
        FieldElement var_power = FieldConstants::ONE;
        for (size_t i = 0; i < degree; ++i) {
            var_power *= point[var_index];
        }
        result *= var_power;
    }
    
    return result;
}

std::string Monomial::to_string() const {
    if (variables.empty()) {
        return "1";
    }
    
    std::stringstream ss;
    bool first = true;
    for (const auto& [var, degree] : variables) {
        if (!first) ss << "*";
        ss << "x_" << var;
        if (degree > 1) {
            ss << "^" << degree;
        }
        first = false;
    }
    return ss.str();
}

// MonomialHash implementation
size_t MonomialHash::operator()(const Monomial& m) const {
    size_t hash = 0;
    for (const auto& [var, degree] : m.variables) {
        hash ^= std::hash<size_t>{}(var) ^ (std::hash<size_t>{}(degree) << 1);
    }
    return hash;
}

// MultivariatePolynomial implementation
MultivariatePolynomial::MultivariatePolynomial() : num_variables(0) {}

MultivariatePolynomial::MultivariatePolynomial(size_t num_vars) : num_variables(num_vars) {}

MultivariatePolynomial::MultivariatePolynomial(
    const std::unordered_map<Monomial, FieldElement, MonomialHash>& terms, 
    size_t num_vars) 
    : terms(terms), num_variables(num_vars) {
    cleanup();
}

MultivariatePolynomial MultivariatePolynomial::zero(size_t num_vars) {
    return MultivariatePolynomial(num_vars);
}

MultivariatePolynomial MultivariatePolynomial::one(size_t num_vars) {
    MultivariatePolynomial result(num_vars);
    result.set_coefficient(Monomial(), Poseidon::FieldConstants::ONE);
    return result;
}

MultivariatePolynomial MultivariatePolynomial::variable(size_t var_index, size_t num_vars) {
    MultivariatePolynomial result(num_vars);
    std::map<size_t, size_t> var_map;
    var_map[var_index] = 1;
    result.set_coefficient(Monomial(var_map), Poseidon::FieldConstants::ONE);
    return result;
}

MultivariatePolynomial MultivariatePolynomial::constant(const FieldElement& coeff, size_t num_vars) {
    MultivariatePolynomial result(num_vars);
    result.set_coefficient(Monomial(), coeff);
    return result;
}

void MultivariatePolynomial::set_coefficient(const Monomial& monomial, const FieldElement& coeff) {
    if (coeff.is_zero()) {
        terms.erase(monomial);
    } else {
        terms[monomial] = coeff;
    }
}

FieldElement MultivariatePolynomial::get_coefficient(const Monomial& monomial) const {
    auto it = terms.find(monomial);
    if (it != terms.end()) {
        return it->second;
    }
    return Poseidon::FieldConstants::ZERO;
}

void MultivariatePolynomial::add_term(const Monomial& monomial, const FieldElement& coeff) {
    if (coeff.is_zero()) return;
    
    auto it = terms.find(monomial);
    if (it != terms.end()) {
        it->second += coeff;
        if (it->second.is_zero()) {
            terms.erase(it);
        }
    } else {
        terms[monomial] = coeff;
    }
}

size_t MultivariatePolynomial::total_degree() const {
    size_t max_degree = 0;
    for (const auto& [monomial, coeff] : terms) {
        max_degree = std::max(max_degree, monomial.total_degree());
    }
    return max_degree;
}

bool MultivariatePolynomial::is_zero() const {
    return terms.empty();
}

MultivariatePolynomial MultivariatePolynomial::operator+(const MultivariatePolynomial& other) const {
    if (num_variables != other.num_variables) {
        throw std::invalid_argument("Cannot add polynomials with different number of variables");
    }
    
    MultivariatePolynomial result(num_variables);
    result.terms = terms;
    
    for (const auto& [monomial, coeff] : other.terms) {
        result.add_term(monomial, coeff);
    }
    
    return result;
}

MultivariatePolynomial MultivariatePolynomial::operator-(const MultivariatePolynomial& other) const {
    if (num_variables != other.num_variables) {
        throw std::invalid_argument("Cannot subtract polynomials with different number of variables");
    }
    
    MultivariatePolynomial result(num_variables);
    result.terms = terms;
    
    for (const auto& [monomial, coeff] : other.terms) {
        result.add_term(monomial, Poseidon::FieldConstants::ZERO - coeff);
    }
    
    return result;
}

MultivariatePolynomial MultivariatePolynomial::operator*(const MultivariatePolynomial& other) const {
    if (num_variables != other.num_variables) {
        throw std::invalid_argument("Cannot multiply polynomials with different number of variables");
    }
    
    MultivariatePolynomial result(num_variables);
    
    for (const auto& [monomial1, coeff1] : terms) {
        for (const auto& [monomial2, coeff2] : other.terms) {
            Monomial new_monomial = monomial1 * monomial2;
            FieldElement new_coeff = coeff1 * coeff2;
            result.add_term(new_monomial, new_coeff);
        }
    }
    
    return result;
}

MultivariatePolynomial MultivariatePolynomial::operator*(const FieldElement& scalar) const {
    if (scalar.is_zero()) {
        return zero(num_variables);
    }
    
    MultivariatePolynomial result(num_variables);
    
    for (const auto& [monomial, coeff] : terms) {
        result.terms[monomial] = coeff * scalar;
    }
    
    return result;
}

MultivariatePolynomial& MultivariatePolynomial::operator+=(const MultivariatePolynomial& other) {
    *this = *this + other;
    return *this;
}

MultivariatePolynomial& MultivariatePolynomial::operator-=(const MultivariatePolynomial& other) {
    *this = *this - other;
    return *this;
}

MultivariatePolynomial& MultivariatePolynomial::operator*=(const MultivariatePolynomial& other) {
    *this = *this * other;
    return *this;
}

MultivariatePolynomial& MultivariatePolynomial::operator*=(const FieldElement& scalar) {
    *this = *this * scalar;
    return *this;
}

FieldElement MultivariatePolynomial::evaluate(const std::vector<FieldElement>& point) const {
    using namespace Poseidon;
    
    FieldElement result = FieldConstants::ZERO;
    
    for (const auto& [monomial, coeff] : terms) {
        FieldElement term_value = coeff * monomial.evaluate(point);
        result += term_value;
    }
    
    return result;
}

MultivariatePolynomial MultivariatePolynomial::partial_evaluate(
    const std::map<size_t, FieldElement>& assignments) const {
    
    MultivariatePolynomial result(num_variables);
    
    for (const auto& [monomial, coeff] : terms) {
        // Create new monomial by removing assigned variables
        std::map<size_t, size_t> new_vars;
        FieldElement eval_coeff = coeff;
        
        for (const auto& [var, degree] : monomial.variables) {
            auto assignment_it = assignments.find(var);
            if (assignment_it != assignments.end()) {
                // Variable is assigned, evaluate it
                FieldElement var_power = Poseidon::FieldConstants::ONE;
                for (size_t i = 0; i < degree; ++i) {
                    var_power *= assignment_it->second;
                }
                eval_coeff *= var_power;
            } else {
                // Variable is not assigned, keep it
                new_vars[var] = degree;
            }
        }
        
        Monomial new_monomial(new_vars);
        result.add_term(new_monomial, eval_coeff);
    }
    
    return result;
}

std::vector<FieldElement> MultivariatePolynomial::evaluate_partial_to_univariate(
    const std::vector<FieldElement>& fixed_values, size_t max_degree) const {
    
    if (fixed_values.size() >= num_variables) {
        throw std::invalid_argument("Too many fixed values");
    }
    
    size_t free_var = fixed_values.size();
    
    // Determine the maximum degree if not provided
    if (max_degree == 0) {
        for (const auto& [monomial, coeff] : terms) {
            auto it = monomial.variables.find(free_var);
            if (it != monomial.variables.end()) {
                max_degree = std::max(max_degree, it->second);
            }
        }
    }
    
    std::vector<FieldElement> coefficients(max_degree + 1, Poseidon::FieldConstants::ZERO);
    
    for (const auto& [monomial, coeff] : terms) {
        // Evaluate all fixed variables
        FieldElement eval_coeff = coeff;
        size_t free_var_degree = 0;
        
        for (const auto& [var, degree] : monomial.variables) {
            if (var < fixed_values.size()) {
                // Fixed variable
                FieldElement var_power = Poseidon::FieldConstants::ONE;
                for (size_t i = 0; i < degree; ++i) {
                    var_power *= fixed_values[var];
                }
                eval_coeff *= var_power;
            } else if (var == free_var) {
                // Free variable
                free_var_degree = degree;
            } else {
                // Variable beyond our scope, treat as constant 1
                // This shouldn't happen in well-formed calls
            }
        }
        
        if (free_var_degree <= max_degree) {
            coefficients[free_var_degree] += eval_coeff;
        }
    }
    
    return coefficients;
}

std::string MultivariatePolynomial::to_string() const {
    if (terms.empty()) {
        return "0";
    }
    
    std::stringstream ss;
    bool first = true;
    
    for (const auto& [monomial, coeff] : terms) {
        if (!first) {
            ss << " + ";
        }
        
        // Show coefficient if it's not 1 or if monomial is constant
        if (!coeff.operator==(Poseidon::FieldConstants::ONE) || monomial.variables.empty()) {
            ss << coeff.to_dec();
            if (!monomial.variables.empty()) {
                ss << "*";
            }
        }
        
        if (!monomial.variables.empty()) {
            ss << monomial.to_string();
        }
        
        first = false;
    }
    
    return ss.str();
}

void MultivariatePolynomial::cleanup() {
    auto it = terms.begin();
    while (it != terms.end()) {
        if (it->second.is_zero()) {
            it = terms.erase(it);
        } else {
            ++it;
        }
    }
}

// Utility functions implementation
namespace Utils {

MultivariatePolynomial random_polynomial(size_t num_vars, size_t max_degree) {
    MultivariatePolynomial result(num_vars);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> degree_dist(0, max_degree);
    std::uniform_int_distribution<size_t> var_dist(0, num_vars - 1);
    std::uniform_int_distribution<size_t> num_terms_dist(1, 20);
    
    size_t num_terms = num_terms_dist(gen);
    
    for (size_t i = 0; i < num_terms; ++i) {
        std::map<size_t, size_t> vars;
        
        // Add random variables with random degrees
        size_t term_degree = degree_dist(gen);
        while (term_degree > 0) {
            size_t var = var_dist(gen);
            size_t deg = std::min(term_degree, degree_dist(gen) + 1);
            vars[var] += deg;
            term_degree -= deg;
        }
        
        Monomial monomial(vars);
        FieldElement coeff = Poseidon::FieldArithmetic::random();
        result.add_term(monomial, coeff);
    }
    
    return result;
}

MultivariatePolynomial boolean_constraint(size_t num_vars) {
    MultivariatePolynomial result = MultivariatePolynomial::one(num_vars);
    
    for (size_t i = 0; i < num_vars; ++i) {
        // For each variable x_i, multiply by (x_i * (1 - x_i))
        // This is 0 when x_i = 0 or x_i = 1
        MultivariatePolynomial var = MultivariatePolynomial::variable(i, num_vars);
        MultivariatePolynomial one_minus_var = MultivariatePolynomial::one(num_vars) - var;
        MultivariatePolynomial constraint = var * one_minus_var;
        result *= constraint;
    }
    
    return result;
}

MultivariatePolynomial multilinear_extension(
    const std::vector<FieldElement>& values, size_t num_vars) {
    
    if (values.size() != (1ULL << num_vars)) {
        throw std::invalid_argument("Values size must be 2^num_vars");
    }
    
    MultivariatePolynomial result = MultivariatePolynomial::zero(num_vars);
    
    // For each point in the Boolean hypercube
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i].is_zero()) continue;
        
        // Create the Lagrange basis polynomial for this point
        MultivariatePolynomial basis = MultivariatePolynomial::one(num_vars);
        
        for (size_t j = 0; j < num_vars; ++j) {
            bool bit = (i >> j) & 1;
            MultivariatePolynomial var = MultivariatePolynomial::variable(j, num_vars);
            
            if (bit) {
                // x_j
                basis *= var;
            } else {
                // (1 - x_j)
                basis *= (MultivariatePolynomial::one(num_vars) - var);
            }
        }
        
        result += basis * values[i];
    }
    
    return result;
}

} // namespace Utils

} // namespace MVPolynomial 