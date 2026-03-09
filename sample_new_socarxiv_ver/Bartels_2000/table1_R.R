# Run Table 1 probit in R (glm with probit link)
library(anesr)
data(timeseries_cum)
df <- timeseries_cum

pres_years <- c(1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996)

df <- df[df$VCF0004 %in% pres_years, ]
df <- df[df$VCF0704a %in% c(1, 2), ]
df$vote_rep <- as.integer(df$VCF0704a == 2)
df <- df[df$VCF0301 %in% 1:7, ]

# Construct symmetric variables
df$strong <- 0
df$strong[df$VCF0301 == 7] <- 1
df$strong[df$VCF0301 == 1] <- -1

df$weak <- 0
df$weak[df$VCF0301 == 6] <- 1
df$weak[df$VCF0301 == 2] <- -1

df$leaning <- 0
df$leaning[df$VCF0301 == 5] <- 1
df$leaning[df$VCF0301 == 3] <- -1

cat(sprintf("%-6s %-5s %-15s %-15s %-15s %-15s %-10s %-6s\n",
            "Year", "N", "Strong", "Weak", "Leaners", "Intercept", "LogLik", "R2"))
cat(paste(rep("-", 90), collapse=""), "\n")

for (yr in pres_years) {
  d <- df[df$VCF0004 == yr, ]
  n <- nrow(d)
  m <- glm(vote_rep ~ strong + weak + leaning, data=d, family=binomial(link="probit"))
  s <- summary(m)
  coefs <- coef(m)
  ses <- s$coefficients[, "Std. Error"]
  ll <- logLik(m)[1]
  null_ll <- logLik(update(m, . ~ 1))[1]
  pseudo_r2 <- 1 - ll/null_ll

  cat(sprintf("%d   %d  %6.3f (%5.3f)  %6.3f (%5.3f)  %6.3f (%5.3f)  %6.3f (%5.3f)  %8.1f  %.2f\n",
              yr, n,
              coefs["strong"], ses["strong"],
              coefs["weak"], ses["weak"],
              coefs["leaning"], ses["leaning"],
              coefs["(Intercept)"], ses["(Intercept)"],
              ll, pseudo_r2))
}
