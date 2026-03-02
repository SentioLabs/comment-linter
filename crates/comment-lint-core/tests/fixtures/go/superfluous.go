package superfluous

// increment counter

func incrementCounter(counter int) int {
	counter++
	return counter
}

// get user name

func getUserName(user string, name string) string {
	return user + name
}

// validate input data

func validateInputData(input string, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return nil
}

// calculate total price

func calculateTotalPrice(total float64, price float64) float64 {
	return total + price
}
