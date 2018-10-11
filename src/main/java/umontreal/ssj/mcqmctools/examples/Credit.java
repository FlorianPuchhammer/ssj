package umontreal.ssj.mcqmctools.examples;

public class Credit {
	private double amount;
	private int rating; //AAA==0, AA==1,...,CCC==6
	private int duration;//in years (1--9)
	private double coupon; //in %
	private int securityClass; // SeniorSecured == 0, SeniorUnsecured == 1, ..., JuniorSubordinated == 4
	private int sector; 
	
	public Credit(double amount, int rating, int duration, double coupon, int securityClass, int sector) {
		this.setAmount(amount);
		this.setRating(rating);
		this.setDuration(duration);
		this.setCoupon(coupon);
		this.setSecurityClass(securityClass);
		this.setSector(sector);
	}
	


	/**
	 * @return the amount
	 */
	public double getAmount() {
		return amount;
	}

	/**
	 * @param amount the amount to set
	 */
	public void setAmount(double amount) {
		this.amount = amount;
	}

	/**
	 * @return the rating
	 */
	public int getRating() {
		return rating;
	}

	/**
	 * @param rating the rating to set
	 */
	public void setRating(int rating) {
		this.rating = rating;
	}

	/**
	 * @return the duration
	 */
	public int getDuration() {
		return duration;
	}

	/**
	 * @param duration the duration to set
	 */
	public void setDuration(int duration) {
		this.duration = duration;
	}

	/**
	 * @return the coupon
	 */
	public double getCoupon() {
		return coupon;
	}

	/**
	 * @param coupon the coupon to set
	 */
	public void setCoupon(double coupon) {
		this.coupon = coupon;
	}

	/**
	 * @return the securityClass
	 */
	public int getSecurityClass() {
		return securityClass;
	}

	/**
	 * @param securityClass the securityClass to set
	 */
	public void setSecurityClass(int securityClass) {
		this.securityClass = securityClass;
	}

	/**
	 * @return the sector
	 */
	public int getSector() {
		return sector;
	}

	/**
	 * @param sector the sector to set
	 */
	public void setSector(int sector) {
		this.sector = sector;
	}
}
