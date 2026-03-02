// get user name
fn get_user_name(user: &str) -> String {
    user.to_string()
}

// increment counter value
fn increment_counter_value(counter: &mut i32, value: i32) {
    *counter += value;
}

// validate input data
fn validate_input_data(input: &str, data: &[u8]) -> bool {
    !input.is_empty() && !data.is_empty()
}

// calculate total price
fn calculate_total_price(total: f64, price: f64) -> f64 {
    total + price
}
