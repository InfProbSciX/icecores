data {
    int n;  // num data
    int s;  // num states per year
    int num_years;
    vector[n] depth;  // depth data
    vector[n] y;  // concentration data
    vector[s * num_years] initial_probs;
}
transformed data {
    int n_st = s * num_years;  // total number of states
    vector[s] year_fractions;
    year_fractions = cumulative_sum(rep_vector(1.0/s, s));
    simplex[n_st] rho = initial_probs + 1e-10;
    rho = rho/sum(rho);
}