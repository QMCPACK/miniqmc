#ifndef QMCPLUSPLUS_DEREFERENCE_ITERATORS_HPP
#define QMCPLUSPLUS_DEREFERENCE_ITERATORS_HPP


namespace qmcplusplus
{

/** Iterate over the unique pointer vectors.
 *  Use with std::algorithms
 */
template<class BaseIterator>
class DereferenceIterator : public BaseIterator
{
public:
  using value_type = typename BaseIterator::value_type::element_type;
  using pointer    = value_type*;
  using reference  = value_type&;

  DereferenceIterator(const BaseIterator& other) : BaseIterator(other) {}

  reference operator*() const { return *(this->BaseIterator::operator*()); }
  pointer operator->() const { return this->BaseIterator::operator*().get(); }
  // reference operator++() const {return *(this->BaseIterator::operator++()); }
  reference operator[](size_t n) const { return *(this->BaseIterator::operator[](n)); }
};

template<typename Iterator>
DereferenceIterator<Iterator> dereference_iterator(Iterator t)
{
  return DereferenceIterator<Iterator>(t);
}

}

#endif
