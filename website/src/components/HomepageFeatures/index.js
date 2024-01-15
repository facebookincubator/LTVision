import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: "Customers' Purchase Patterns Insights & pLTV opportunity size",
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        <li>Visualize customer’s purchase pattern</li>
        <li>Assess revenue contribution of high, medium, low value customers</li>
        <li>Evaluate opportunity size of maximizing acquisition of high-value customers</li>
      </>
    ),
  },
  {
    title: 'Structured Approach to pLTV Modeling',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        <li>Establish the relationship with short term revenue of customers versus long term revenue of customers</li>
        <li>Structured approach to decide modeling parameters of when to predict and how long to predict to maximize the opportunity with pLTV</li>
      </>
    ),
  },
  {
    title: 'Aligning pLTV Model Outputs to Marketing Strategies',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        <li>Understand what are different marketing strategies / optimization techniques you can use with pLTV model outputs </li>
        <li>Organize model outputs to use for marketing optimizations</li>
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--left padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
