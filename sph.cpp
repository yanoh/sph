#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

#define _USE_MATH_DEFINES
#include <cmath>

template <typename T>
class numeric {
public:
    T operator + () const {
        const T& self = static_cast<const T&>(*this);
        return self;
    }
    T operator - () const {
        const T& self = static_cast<const T&>(*this);
        return -1 * self;
    }

    template <typename U> T& operator = (const U& rhs) {
        T& self = static_cast<T&>(*this);
        self = T(rhs);
        return self;
    }

    template <typename U> T& operator += (const U& rhs) {
        T& self = static_cast<T&>(*this);
        return self = self + rhs;
    }
    template <typename U> T& operator -= (const U& rhs) {
        T& self = static_cast<T&>(*this);
        return self = self - rhs;
    }
    template <typename U> T& operator *= (const U& rhs) {
        T& self = static_cast<T&>(*this);
        return self = self * rhs;
    }
    template <typename U> T& operator /= (const U& rhs) {
        T& self = static_cast<T&>(*this);
        return self = self / rhs;
    }
};

template <typename T, typename U>
class comparable {
    friend inline bool operator == (const T& lhs, const U& rhs) { return lhs == T(rhs); }
    friend inline bool operator == (const U& lhs, const T& rhs) { return T(lhs) == rhs; }
    friend inline bool operator != (const T& lhs, const U& rhs) { return lhs != T(rhs); }
    friend inline bool operator != (const U& lhs, const T& rhs) { return T(lhs) != rhs; }
    friend inline bool operator <  (const T& lhs, const U& rhs) { return lhs <  T(rhs); }
    friend inline bool operator <  (const U& lhs, const T& rhs) { return T(lhs) <  rhs; }
    friend inline bool operator >  (const T& lhs, const U& rhs) { return lhs >  T(rhs); }
    friend inline bool operator >  (const U& lhs, const T& rhs) { return T(lhs) >  rhs; }
    friend inline bool operator <= (const T& lhs, const U& rhs) { return lhs <= T(rhs); }
    friend inline bool operator <= (const U& lhs, const T& rhs) { return T(lhs) <= rhs; }
    friend inline bool operator >= (const T& lhs, const U& rhs) { return lhs >= T(rhs); }
    friend inline bool operator >= (const U& lhs, const T& rhs) { return T(lhs) >= rhs; }
};

template <typename T, typename U>
class arithmetic {
    friend inline T operator + (const T& lhs, const U& rhs) { return lhs + T(rhs); }
    friend inline T operator + (const U& lhs, const T& rhs) { return T(lhs) + rhs; }
    friend inline T operator - (const T& lhs, const U& rhs) { return lhs - T(rhs); }
    friend inline T operator - (const U& lhs, const T& rhs) { return T(lhs) - rhs; }
    friend inline T operator * (const T& lhs, const U& rhs) { return lhs * T(rhs); }
    friend inline T operator * (const U& lhs, const T& rhs) { return T(lhs) * rhs; }
    friend inline T operator / (const T& lhs, const U& rhs) { return lhs / T(rhs); }
    friend inline T operator / (const U& lhs, const T& rhs) { return T(lhs) / rhs; }
    friend inline T operator % (const T& lhs, const U& rhs) { return lhs % T(rhs); }
    friend inline T operator % (const U& lhs, const T& rhs) { return T(lhs) % rhs; }
    friend inline T operator ^ (const T& lhs, const U& rhs) { return lhs ^ T(rhs); }
    friend inline T operator ^ (const U& lhs, const T& rhs) { return T(lhs) ^ rhs; }
};

class real : public numeric<real>,
             comparable<real, double>, comparable<real, int>,
             arithmetic<real, double>, arithmetic<real, int> {
public:
    using value_type = double;
    real(value_type val = 0) : val_(val) {}

    operator int () const { return static_cast<int>(round().value()); }
    operator value_type () const { return val_; }
    value_type value() const { return val_; }

    real abs() const { return real(std::abs(val_)); }
    real sqrt() const { return real(std::sqrt(val_)); }
    real ceil() const { return real(std::ceil(val_)); }
    real floor() const { return real(std::floor(val_)); }
    real round() const { return *this < 0 ? ceil() : floor(); }
    real pow(const real& exp) const { return real(std::pow(val_, exp)); }

    bool is_nan() const { return std::isnan(val_); }

private:
    value_type val_;
};

inline std::ostream& operator << (std::ostream& os, const real& num) {
    return os << std::setprecision(9) << std::fixed << num.value();
}

inline real operator + (const real& lhs, const real& rhs) { return real(lhs.value() + rhs.value()); }
inline real operator - (const real& lhs, const real& rhs) { return real(lhs.value() - rhs.value()); }
inline real operator * (const real& lhs, const real& rhs) { return real(lhs.value() * rhs.value()); }
inline real operator / (const real& lhs, const real& rhs) { return real(lhs.value() / rhs.value()); }

inline bool almost_equal(const real& lhs, const real& rhs) {
    assert(!std::isunordered(lhs.value(), rhs.value()));
    return std::abs(lhs.value() - rhs.value()) < 1e-9;
}

inline bool operator <  (const real& lhs, const real& rhs) {
    return std::isless(lhs.value(), rhs.value());
}
inline bool operator >  (const real& lhs, const real& rhs) {
    return std::isgreater(lhs.value(), rhs.value());
}
inline bool operator == (const real& lhs, const real& rhs) {
    return !std::isunordered(lhs.value(), rhs.value()) &&  almost_equal(lhs, rhs);
}
inline bool operator != (const real& lhs, const real& rhs) {
    return !std::isunordered(lhs.value(), rhs.value()) && !almost_equal(lhs, rhs);
}
inline bool operator <= (const real& lhs, const real& rhs) {
    return std::islessequal(lhs.value(), rhs.value());
}
inline bool operator >= (const real& lhs, const real& rhs) {
    return std::isgreaterequal(lhs.value(), rhs.value());
}

template <typename T>
class vector : public numeric<vector<T>>, arithmetic<vector<T>, T> {
public:
    vector() {}
    vector(const T& x_, const T& y_, const T& z_) : x(x_), y(y_), z(z_) {}
    vector(const T& v) : x(v), y(v), z(v) {}

    vector& operator = (const T& v) {
        x = y = z = v;
        return *this;
    }

    T len() const { return pow2len().sqrt(); }
    T pow2len() const { return x*x + y*y + z*z; }
    T dot(const vector& rhs) const { return x*rhs.x + y*rhs.y + z*rhs.z; }

    T x, y, z;
};

template <typename T>
class cube {
public:
    cube(const vector<real>& l, const vector<real>& u) : lower_limit_(l), upper_limit_(u) {}

    const vector<real>& lower_limit() const { return lower_limit_; }
    const vector<real>& upper_limit() const { return upper_limit_; }

    real width() const { return upper_limit_.x - lower_limit_.x; }
    real height() const { return upper_limit_.y - lower_limit_.y; }
    real depth() const { return upper_limit_.z - lower_limit_.z; }

private:
    vector<real> lower_limit_, upper_limit_;
};

template <typename T> inline std::ostream& operator << (std::ostream& os, const vector<T>& vec) {
    return os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
}

template <typename T> inline vector<T> operator + (const vector<T>& lhs, const vector<T>& rhs) {
    return vector<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
template <typename T> inline vector<T> operator - (const vector<T>& lhs, const vector<T>& rhs) {
    return vector<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
template <typename T> inline vector<T> operator * (const vector<T>& lhs, const vector<T>& rhs) {
    return vector<T>(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}
template <typename T> inline vector<T> operator / (const vector<T>& lhs, const vector<T>& rhs) {
    return vector<T>(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

template <typename T> inline bool operator == (const vector<T>& lhs, const vector<T>& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
template <typename T> inline bool operator != (const vector<T>& lhs, const vector<T>& rhs) {
    return !(lhs == rhs);
}

#include PARAM

const real DT           = 1e-3;
const real H            = 1e-2;
const real POLY6_KERNEL = 315 / (M_PI * H.pow(9) * 64);
const real SPIKY_KERNEL = -45 / (M_PI * H.pow(6));
const real KERNEL_LAP   =  45 / (M_PI * H.pow(6));

class particle {
public:
    particle() {}
    particle(const real& x, const real& y, const real& z) : pos(x, y, z) {}

    vector<real> pos, vel, frc;
    real rho, prs;
};

class particles {
public:
    particles() : bounds_(vector<real>(0, 0, 0), vector<real>(50, 50, 50)) {}

    particles& refill1() {
        const real d = DISTRIB / SCALE * 0.95;
        for (real x = 10 + d; x <= 30 - d; x += d) {
            for (real y = 20 + d; y <= 40 - d; y += d) {
                for (real z = 25 + d; z <= 50 - d; z += d) {
                    prts_.push_back(particle(x, y, z));
                }
            }
        }
        return *this;
    }

    particles& refill2() {
        const real d = DISTRIB / SCALE * 0.95;
        for (real x = 20 + d; x <= 40 - d; x += d) {
            for (real y = 10 + d; y <= 30 - d; y += d) {
                for (real z = 25 + d; z <= 50 - d; z += d) {
                    prts_.push_back(particle(x, y, z));
                }
            }
        }
        return *this;
    }

    std::size_t size() const { return prts_.size(); }
    const cube<real>& bounds() const { return bounds_; }

    void each(std::function<void(const particle&)> fn) const {
        std::for_each(prts_.begin(), prts_.end(), fn);
    }
    void each(std::function<void(particle&)> fn) {
        std::for_each(prts_.begin(), prts_.end(), fn);
    }

    particles& next() {
        neighborhood nb(*this);
        compute_density_and_pressure(nb);
        compute_force(nb);
        advance();
        return *this;
    }

    void save(int frm) const {
        std::ostringstream ss;
        ss << "sph-" << std::setw(4) << std::setfill('0') << frm << ".mat";
        std::ofstream fs(ss.str());
        fs << *this;
        fs.close();
    }

private:
    cube<real> bounds_;
    std::vector<particle> prts_;

    friend inline std::ostream& operator << (std::ostream& os, const particles& prts) {
        os << "# name: particles\n"
           << "# type: matrix\n"
           << "# rows: 3\n"
           << "# columns: " << prts.size() << '\n';
        prts.each([&](const particle& prt) { os << ' ' << prt.pos.x; });
        os << '\n';
        prts.each([&](const particle& prt) { os << ' ' << prt.pos.y; });
        os << '\n';
        prts.each([&](const particle& prt) { os << ' ' << prt.pos.z; });
        os << '\n';
        return os;
    }

    class neighborhood {
    public:
        using neighbors = std::vector<particle*>;

        neighborhood(particles& prts) : prts_(prts) {
            prts.each([&](particle& prt) { graph_[index_of(prt.pos)].push_back(&prt); });
        }

        neighbors neighbors_of(particle& prt) {
            neighbors nbs;
            const real d = H / SCALE;
            for (int x = -1; x < 2; ++x) {
                for (int y = -1; y < 2; ++y) {
                    for (int z = -1; z < 2; ++z) {
                        const vector<real> pos(prt.pos.x + x*d, prt.pos.y + y*d, prt.pos.z + z*d);
                        entry_type& ent = graph_[index_of(pos)];
                        std::copy(ent.begin(), ent.end(), std::back_inserter(nbs));
                    }
                }
            }
            return nbs;
        }
    
    private:
        int index_of(const vector<real>& pos) const {
            const real d = H / SCALE;
            const int x  = (pos.x - prts_.bounds().lower_limit().x) / d;
            const int y  = (pos.y - prts_.bounds().lower_limit().y) / d;
            const int z  = (pos.z - prts_.bounds().lower_limit().z) / d;
            const int mx = prts_.bounds().width()  / d;
            const int my = prts_.bounds().height() / d;
            return x + y * mx + z * mx * my;
        }

        particles& prts_;
        using entry_type = std::vector<particle*>;
        using graph_type = std::map<int, entry_type>;
        graph_type graph_;
    };

    void compute_density_and_pressure(neighborhood& nb) {
        std::for_each(prts_.begin(), prts_.end(), [&](particle& prt) {
            real sum;
            neighborhood::neighbors n = nb.neighbors_of(prt);
            std::for_each(n.begin(), n.end(), [&](particle* prt_j) {
                const vector<real> dr = (prt.pos - prt_j->pos) * SCALE;
                const real H2 = H*H;
                const real r2 = dr.pow2len();
                if (r2 < H2) {
                    sum += (H2 - r2).pow(3);
                }
            });
            prt.rho = sum * MASS * POLY6_KERNEL;
            prt.prs = (prt.rho - DENS) * 3;
            prt.rho = 1 / prt.rho;
        });
    }

    void compute_force(neighborhood& nb) {
        const real visc_term = KERNEL_LAP * VISC;
        std::for_each(prts_.begin(), prts_.end(), [&](particle& prt) {
            prt.frc = 0;
            neighborhood::neighbors n = nb.neighbors_of(prt);
            std::for_each(n.begin(), n.end(), [&](particle* prt_j) {
                if (prt.pos != prt_j->pos) {
                    const vector<real> dr = (prt.pos - prt_j->pos) * SCALE;
                    const real r = dr.len();
                    if (r < H) {
                        const real q = H - r;
                        const real prs_term = -0.5 * q * SPIKY_KERNEL * (prt.prs + prt_j->prs) / r;
                        const vector<real> cur_frc =
                            (prs_term * dr + visc_term * (prt_j->vel - prt.vel)) * (q * prt.rho * prt_j->rho);
                        prt.frc += cur_frc;
                    }
                }
            });
        });
    }

    void advance() {
        const vector<real> gacc(0.0, 0.0, -9.8);
        std::for_each(prts_.begin(), prts_.end(), [&](particle& prt) {
            vector<real> acc = prt.frc * MASS;
            const real vel = acc.pow2len();
            if (vel > VLIMIT*VLIMIT) acc *= VLIMIT / vel.sqrt();
            accelerate(acc, prt, [](const vector<real>& v) { return v.x; }, vector<real>(1, 0, 0));
            accelerate(acc, prt, [](const vector<real>& v) { return v.y; }, vector<real>(0, 1, 0));
            accelerate(acc, prt, [](const vector<real>& v) { return v.z; }, vector<real>(0, 0, 1));
            acc += gacc;
            prt.vel += acc * DT;
            prt.pos += prt.vel * DT / SCALE;
        });
    }

    template <typename S>
    void accelerate(vector<real>& acc, const particle& prt, S sel, const vector<real>& norm) {
        const real ld = 2*RADIUS - (sel(prt.pos) - sel(bounds().lower_limit())) * SCALE;
        if (ld > 1e-6) {
            acc += (1e+4 * ld - RESIST * norm.dot(prt.vel)) * norm;
        }
        const real ud = 2*RADIUS - (sel(bounds().upper_limit()) - sel(prt.pos)) * SCALE;
        if (ud > 1e-6) {
            acc += (1e+4 * ud - RESIST * -norm.dot(prt.vel)) * -norm;
        }
    }
};

int main() {
    int frm = 0;

    particles prts;
    prts.refill1().save(frm++);

    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
    prts.refill2();
    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
    prts.refill1();
    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
    prts.refill2();
    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
    prts.refill1();
    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
    prts.refill2();
    for (int i = 0; i < 600; ++i) prts.next().save(frm++);
}

// vim:tw=109:cc=+2:
